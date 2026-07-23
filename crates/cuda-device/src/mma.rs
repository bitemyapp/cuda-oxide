/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix multiply-accumulate intrinsics.

use crate::cusimd::CuSimd;

/// Execute one warp-cooperative `m16n8k32` unsigned-byte MMA.
///
/// The four `a` and two `b` fragments use the PTX lane layout for
/// `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32`. The four accumulator
/// fragments are both the C input and the returned D output.
///
/// # Safety
///
/// All 32 lanes in the warp must execute this operation convergently with
/// correctly packed fragments.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mma_m16n8k32_u8(
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    b0: u32,
    b1: u32,
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
) -> CuSimd<u32, 4> {
    let _ = (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    unreachable!("mma_m16n8k32_u8 called outside CUDA kernel context")
}
