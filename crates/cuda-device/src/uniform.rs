/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-uniform read-only global-memory loads.

/// Load one `u64` from a read-only global address shared by every active lane in the warp.
///
/// This lowers to PTX `ldu.global.u64`, which lets the target use its uniform-load path instead of
/// issuing an ordinary per-lane global load.
///
/// # Safety
///
/// Every active lane in the warp that executes this call must provide the same valid, aligned
/// global-memory address. The pointed-to value must not be mutated for the duration of the kernel.
#[inline(never)]
pub unsafe fn load_u64(ptr: *const u64) -> u64 {
    let _ = ptr;
    unreachable!("uniform::load_u64 called outside CUDA kernel context")
}
