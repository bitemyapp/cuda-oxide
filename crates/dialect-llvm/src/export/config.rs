/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend configuration traits and built-in implementations.

/// Minimal data layout for PTX mode (default behavior).
pub(super) const NVPTX_DATALAYOUT_PTX: &str = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

/// Full NVPTX data layout for libNVVM/LTOIR mode (Blackwell+, LLVM 20 dialect).
///
/// This matches nvcc's output for sm_100+ and is required for full NVVM compatibility.
pub(super) const NVPTX_DATALAYOUT_FULL: &str = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8";

/// Configuration trait for export backends (PTX, LTOIR, etc.).
///
/// This trait allows different backends to customize IR generation without
/// exposing backend-specific details in the public API.
pub trait ExportBackendConfig {
    /// Data layout string for the target.
    fn datalayout(&self) -> &str;

    /// Whether to emit `@llvm.used` for kernel functions.
    /// This prevents the optimizer from removing "unused" kernels.
    fn emit_llvm_used(&self) -> bool;

    /// Whether to emit `!nvvmir.version` metadata.
    fn emit_nvvmir_version(&self) -> bool;

    /// The version tuple for `!nvvmir.version` metadata.
    /// Format: [major, minor, debug_major, debug_minor]
    fn nvvmir_version(&self) -> [i32; 4];

    /// Whether to emit `!nvvm.annotations` for ALL kernels.
    /// When false, only kernels with special attributes get annotations.
    fn emit_all_kernel_annotations(&self) -> bool;

    /// Whether kernel definitions should use the `ptx_kernel` calling convention.
    fn emit_ptx_kernel_keyword(&self) -> bool;

    /// Whether to emit LLVM-7-style TYPED pointers (`i8*`, `i64 addrspace(3)*`, …) instead of
    /// opaque `ptr`. Required for libNVVM targeting pre-Hopper GPUs (sm_89 / Ada and older):
    /// its legacy IR parser rejects opaque `ptr`. libNVVM's modern (sm_90+) parser accepts BOTH
    /// forms, so typed output is also valid on Blackwell — but production keeps opaque there to
    /// minimize churn; this is opt-in via the pre-Blackwell config. Emitted as `i8*`-uniform
    /// with per-op `bitcast`s to the concrete access type (dataflow-free reconstruction from each
    /// op's own element type — the dialect pointer type is opaque, carrying only address space).
    fn typed_pointers(&self) -> bool {
        false
    }
}

/// Default PTX export configuration.
///
/// Uses minimal settings appropriate for standard PTX generation via llc.
#[derive(Clone, Debug, Default)]
pub struct PtxExportConfig;

impl ExportBackendConfig for PtxExportConfig {
    fn datalayout(&self) -> &str {
        NVPTX_DATALAYOUT_PTX
    }

    fn emit_llvm_used(&self) -> bool {
        false
    }

    fn emit_nvvmir_version(&self) -> bool {
        false
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [0, 0, 0, 0] // Not used in PTX mode
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        false
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        true
    }
}

/// Export configuration for NVVM IR output.
///
/// Emits LLVM IR with full NVVM compatibility:
/// - Full NVPTX datalayout string
/// - `@llvm.used` to prevent kernel optimization
/// - `!nvvm.annotations` for all kernels
/// - `!nvvmir.version` metadata
///
/// This produces IR suitable for consumption by libNVVM (e.g., `nvvmCompileProgram -gen-lto`)
/// or other NVVM-compatible tools.
///
/// Currently supports NVVM 20 dialect (Blackwell+, opaque pointers).
/// NVVM 7 dialect (pre-Blackwell, typed pointers) is not yet supported.
#[derive(Clone, Debug, Default)]
pub struct NvvmExportConfig;

impl ExportBackendConfig for NvvmExportConfig {
    fn datalayout(&self) -> &str {
        NVPTX_DATALAYOUT_FULL
    }

    fn emit_llvm_used(&self) -> bool {
        true
    }

    fn emit_nvvmir_version(&self) -> bool {
        true
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [2, 0, 3, 2] // NVVM IR 2.0, debug 3.2
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        true // Emit annotations for all kernels
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }
}

/// NVVM IR export for pre-Hopper targets (sm_89 / Ada and older): identical to
/// [`NvvmExportConfig`] except it emits LLVM-7-style typed pointers, which libNVVM's legacy
/// parser (used for compute_89 and below) requires. See [`ExportBackendConfig::typed_pointers`].
#[derive(Clone, Debug, Default)]
pub struct NvvmTypedPtrExportConfig;

impl ExportBackendConfig for NvvmTypedPtrExportConfig {
    fn datalayout(&self) -> &str {
        NVPTX_DATALAYOUT_FULL
    }

    fn emit_llvm_used(&self) -> bool {
        true
    }

    fn emit_nvvmir_version(&self) -> bool {
        true
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [2, 0, 3, 2]
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        true
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }

    fn typed_pointers(&self) -> bool {
        true
    }
}
