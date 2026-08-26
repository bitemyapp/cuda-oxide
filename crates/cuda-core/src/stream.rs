/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA stream management (RAII, host callbacks, fork/join).
//!
//! A [`CudaStream`] wraps a `CUstream` handle and ties its lifetime to its
//! parent [`CudaContext`]. Streams are created via
//! [`CudaContext::new_stream`] or [`CudaStream::fork`] and destroyed
//! automatically on [`Drop`].
//!
//! # Ordering model
//!
//! Operations enqueued on the same stream execute in FIFO order. Operations on
//! different streams may overlap. Use [`fork`](CudaStream::fork) /
//! [`join`](CudaStream::join) or explicit events ([`record_event`](CudaStream::record_event),
//! [`wait`](CudaStream::wait)) to establish cross-stream ordering.
//!
//! # Host callbacks
//!
//! [`launch_host_function`](CudaStream::launch_host_function) enqueues a
//! host-side closure that the driver invokes after all preceding stream work
//! completes. This is the primary bridge between CUDA stream completion and
//! Rust `async` futures.

use crate::context::CudaContext;
use crate::device_buffer::DeviceBuffer;
use crate::error::{DriverError, IntoResult};
use crate::event::CudaEvent;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// An RAII wrapper around a `CUstream` handle.
///
/// Holds an `Arc<CudaContext>` to ensure the context outlives the stream.
/// A null `cu_stream` represents the per-context default stream (stream 0).
#[derive(Debug, PartialEq, Eq)]
pub struct CudaStream {
    /// Raw CUDA stream handle. Null for the default stream.
    pub(crate) cu_stream: cuda_bindings::CUstream,
    /// Owning context. Kept alive for the lifetime of this stream.
    pub(crate) ctx: Arc<CudaContext>,
}

/// # Safety
///
/// `CUstream` handles are not thread-local. The CUDA driver permits enqueuing
/// work onto a stream from any thread, provided the owning context is bound
/// (which [`bind_to_thread`](CudaContext::bind_to_thread) ensures).
unsafe impl Send for CudaStream {}
/// See [`Send`] impl.
unsafe impl Sync for CudaStream {}

/// Destroys the underlying `CUstream` on drop and decrements the context's
/// live stream count.
///
/// The default stream (null handle) is never destroyed. Errors during
/// teardown are recorded on the context rather than panicking.
impl Drop for CudaStream {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        if !self.cu_stream.is_null() {
            self.ctx.num_streams.fetch_sub(1, Ordering::Relaxed);
            self.ctx
                .record_err(unsafe { cuda_bindings::cuStreamDestroy_v2(self.cu_stream).result() });
        }
    }
}

impl CudaStream {
    /// Returns the raw `CUstream` handle (null for the default stream).
    pub fn cu_stream(&self) -> cuda_bindings::CUstream {
        self.cu_stream
    }

    /// Returns the parent [`CudaContext`].
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the priority this stream actually runs at.
    ///
    /// Wraps `cuStreamGetPriority`. This is the value to read after
    /// [`CudaContext::new_stream_with_priority`](crate::context::CudaContext::new_stream_with_priority),
    /// which clamps an out-of-range request to the device's range without
    /// reporting it. Lower numbers are higher priorities. The default stream
    /// and any stream from [`CudaContext::new_stream`](crate::context::CudaContext::new_stream)
    /// report `0`.
    pub fn priority(&self) -> Result<i32, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut priority = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuStreamGetPriority(self.cu_stream, priority.as_mut_ptr()).result()?;
            Ok(priority.assume_init())
        }
    }

    /// Blocks the calling thread until all work enqueued on this stream
    /// completes.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuStreamSynchronize(self.cu_stream) }.result()
    }

    /// Applies a persisting-L2 access-policy window over the start of `buffer`.
    ///
    /// The requested byte count is clamped to both the allocation size and the
    /// device's maximum access-policy window. `hit_ratio` must be finite and in
    /// `0.0..=1.0`. The returned value is the window size passed to the driver.
    ///
    /// The buffer must belong to the same CUDA context as this stream. Reserve
    /// persisting-L2 capacity first with
    /// [`CudaContext::set_limit`](crate::context::CudaContext::set_limit) and
    /// [`ContextLimit::PersistingL2CacheSize`](crate::context::ContextLimit::PersistingL2CacheSize).
    pub fn set_persisting_access_policy<T>(
        &self,
        buffer: &DeviceBuffer<T>,
        requested_bytes: usize,
        hit_ratio: f32,
    ) -> Result<usize, DriverError> {
        if self.ctx.as_ref() != buffer.context().as_ref() || !valid_hit_ratio(hit_ratio) {
            return Err(invalid_value());
        }

        let maximum_bytes = self
            .ctx
            .persisting_l2_cache_limits()?
            .max_access_policy_window_bytes;
        let applied_bytes =
            clamp_access_policy_window_bytes(requested_bytes, buffer.num_bytes(), maximum_bytes);
        if applied_bytes == 0 {
            return Err(invalid_value());
        }

        self.set_access_policy_window(
            buffer.cu_deviceptr() as *mut c_void,
            applied_bytes,
            hit_ratio,
            cuda_bindings::CUaccessProperty_enum_CU_ACCESS_PROPERTY_PERSISTING,
            cuda_bindings::CUaccessProperty_enum_CU_ACCESS_PROPERTY_STREAMING,
        )?;
        Ok(applied_bytes)
    }

    /// Removes this stream's access-policy window and restores normal caching.
    pub fn clear_access_policy_window(&self) -> Result<(), DriverError> {
        self.set_access_policy_window(
            std::ptr::null_mut(),
            0,
            0.0,
            cuda_bindings::CUaccessProperty_enum_CU_ACCESS_PROPERTY_NORMAL,
            cuda_bindings::CUaccessProperty_enum_CU_ACCESS_PROPERTY_NORMAL,
        )
    }

    fn set_access_policy_window(
        &self,
        base_ptr: *mut c_void,
        num_bytes: usize,
        hit_ratio: f32,
        hit_property: cuda_bindings::CUaccessProperty,
        miss_property: cuda_bindings::CUaccessProperty,
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        let window = cuda_bindings::CUaccessPolicyWindow {
            base_ptr,
            num_bytes,
            hitRatio: hit_ratio,
            hitProp: hit_property,
            missProp: miss_property,
        };
        let mut value: cuda_bindings::CUstreamAttrValue = unsafe { std::mem::zeroed() };
        unsafe {
            // CUstreamAttrValue is an opaque union in generated CUDA 13
            // bindings. CUaccessPolicyWindow is its first member.
            std::ptr::write(
                std::ptr::addr_of_mut!(value).cast::<cuda_bindings::CUaccessPolicyWindow>(),
                window,
            );
            cuda_bindings::cuStreamSetAttribute(
                self.cu_stream,
                cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                std::ptr::addr_of!(value),
            )
            .result()
        }
    }

    /// Returns `true` when every operation enqueued on this stream has
    /// completed, `false` while any is still in flight. Never blocks.
    ///
    /// Wraps `cuStreamQuery`, mapping `CUDA_SUCCESS` to `Ok(true)` and
    /// `CUDA_ERROR_NOT_READY` to `Ok(false)`, as
    /// [`CudaEvent::query`](crate::event::CudaEvent::query) does for
    /// `cuEventQuery`. Any other code is a real driver error.
    ///
    /// This is the completion half of the async bridge that
    /// [`launch_host_function`](Self::launch_host_function) opens: a
    /// `Future::poll` implementation needs to ask whether work has finished
    /// without parking the executor's thread, which
    /// [`synchronize`](Self::synchronize) cannot answer.
    ///
    /// A sticky error from earlier work on the stream surfaces here rather
    /// than as `Ok(false)`, the same way it would from
    /// [`synchronize`](Self::synchronize).
    ///
    /// On the default stream this reports the completion of every blocking
    /// stream in the context, since the default stream orders against them
    /// all.
    pub fn query(&self) -> Result<bool, DriverError> {
        self.ctx.bind_to_thread()?;
        match unsafe { cuda_bindings::cuStreamQuery(self.cu_stream) } {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(true),
            cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY => Ok(false),
            err => Err(DriverError(err)),
        }
    }

    /// Creates a new non-blocking stream that waits on all prior work in
    /// `self` before executing its own.
    ///
    /// Semantically equivalent to creating a new stream and calling
    /// [`join`](Self::join) on it with `self`, establishing a fork point
    /// in the stream DAG.
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        self.ctx.bind_to_thread()?;
        self.ctx.num_streams.fetch_add(1, Ordering::Relaxed);
        let mut cu_stream = MaybeUninit::uninit();
        let cu_stream = unsafe {
            cuda_bindings::cuStreamCreate(
                cu_stream.as_mut_ptr(),
                cuda_bindings::CUstream_flags_enum_CU_STREAM_NON_BLOCKING,
            )
            .result()?;
            cu_stream.assume_init()
        };
        let stream = Arc::new(CudaStream {
            cu_stream,
            ctx: self.ctx.clone(),
        });
        stream.join(self)?;
        Ok(stream)
    }

    /// Makes `self` wait on all prior work in `other`.
    ///
    /// Records an event on `other` and enqueues a wait on `self`. This is the
    /// join side of the fork/join pattern: after this call, work enqueued on
    /// `self` is guaranteed to observe all side effects of prior work on
    /// `other`.
    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
    }

    /// Records an event on this stream and returns it.
    ///
    /// `flags` defaults to `CU_EVENT_DISABLE_TIMING` when `None`, which is
    /// cheaper than a timing-enabled event. Pass
    /// `Some(CU_EVENT_DEFAULT)` if you need [`CudaEvent::elapsed_ms`].
    pub fn record_event(
        &self,
        flags: Option<cuda_bindings::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let event = self.ctx.new_event(flags)?;
        event.record(self)?;
        Ok(event)
    }

    /// Enqueues a wait on `event` into this stream.
    ///
    /// All work enqueued on `self` after this call will not begin until
    /// `event` has been recorded (i.e., all prior work on the stream that
    /// recorded `event` has completed).
    pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            cuda_bindings::cuStreamWaitEvent(
                self.cu_stream,
                event.cu_event(),
                cuda_bindings::CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
            )
            .result()
        }
    }

    /// Enqueues a host-side callback that the driver invokes after all prior
    /// work on this stream completes.
    ///
    /// This is the bridge between CUDA stream ordering and Rust async: wrap a
    /// `oneshot::Sender::send` or `Waker::wake` in `host_func` to unblock a
    /// future when GPU work finishes.
    ///
    /// `host_func` is boxed, leaked into a raw pointer, and passed as user
    /// data to `cuLaunchHostFunc`. The driver calls
    /// `callback_wrapper` on a driver-internal
    /// thread, which reclaims the box and invokes the closure.
    ///
    /// Panics inside the closure are caught and discarded to prevent unwinding
    /// across the FFI boundary.
    pub fn launch_host_function<F: FnOnce() + Send>(
        &self,
        host_func: F,
    ) -> Result<(), DriverError> {
        let boxed = Box::new(host_func);
        unsafe {
            cuda_bindings::cuLaunchHostFunc(
                self.cu_stream,
                Some(Self::callback_wrapper::<F>),
                Box::into_raw(boxed) as *mut c_void,
            )
            .result()
        }
    }

    /// `extern "C"` trampoline invoked by the CUDA driver on a driver-internal
    /// thread when a host function callback fires.
    ///
    /// Reconstructs the `Box<F>` from the raw pointer and calls the closure.
    /// Panics are caught with `catch_unwind` to prevent unwinding across the
    /// C ABI boundary.
    ///
    /// # Safety
    ///
    /// - `callback` must be a pointer produced by `Box::into_raw(Box::new(f))`
    ///   where `f: F`.
    /// - Must be called exactly once per enqueued callback (double-free
    ///   otherwise).
    unsafe extern "C" fn callback_wrapper<F: FnOnce() + Send>(callback: *mut c_void) {
        let _ = std::panic::catch_unwind(|| {
            let callback: Box<F> = unsafe { Box::from_raw(callback as *mut F) };
            callback();
        });
    }
}

fn invalid_value() -> DriverError {
    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE)
}

fn valid_hit_ratio(hit_ratio: f32) -> bool {
    hit_ratio.is_finite() && (0.0..=1.0).contains(&hit_ratio)
}

fn clamp_access_policy_window_bytes(
    requested_bytes: usize,
    allocation_bytes: usize,
    maximum_bytes: usize,
) -> usize {
    requested_bytes.min(allocation_bytes).min(maximum_bytes)
}

#[cfg(test)]
mod tests {
    use super::{clamp_access_policy_window_bytes, valid_hit_ratio};

    #[test]
    fn access_policy_window_is_clamped_to_every_bound() {
        assert_eq!(clamp_access_policy_window_bytes(32, 64, 128), 32);
        assert_eq!(clamp_access_policy_window_bytes(96, 64, 128), 64);
        assert_eq!(clamp_access_policy_window_bytes(256, 512, 128), 128);
        assert_eq!(clamp_access_policy_window_bytes(1, 0, 128), 0);
    }

    #[test]
    fn access_policy_hit_ratio_must_be_a_probability() {
        assert!(valid_hit_ratio(0.0));
        assert!(valid_hit_ratio(1.0));
        assert!(!valid_hit_ratio(-f32::EPSILON));
        assert!(!valid_hit_ratio(1.0 + f32::EPSILON));
        assert!(!valid_hit_ratio(f32::NAN));
        assert!(!valid_hit_ratio(f32::INFINITY));
    }
}
