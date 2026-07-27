/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer arithmetic intrinsics that require an uninterrupted PTX carry chain.

/// Multiply two residues using the backend's fixed 64-bit modulus.
///
/// The result is a congruent, partially reduced value in `[0, 2^64)`. Callers may feed it back
/// into this intrinsic without canonicalizing; canonicalization is required only at an observable
/// modular-arithmetic boundary.
///
/// cuda-oxide replaces this marker with a 32-bit-limb PTX multiply/reduction carry chain.
#[inline(never)]
pub fn mul_mod_p64_partial(a: u64, b: u64) -> u64 {
    let _ = (a, b);
    unreachable!("mul_mod_p64_partial called outside CUDA kernel context")
}

/// Compute a two-term multiply-accumulate using the fixed 64-bit modulus.
///
/// cuda-oxide replaces this marker with one exact five-limb accumulation and reduction.
#[inline(never)]
pub fn dot2_mod_p64(acc: u64, a0: u64, b0: u64, a1: u64, b1: u64) -> u64 {
    let _ = (acc, a0, b0, a1, b1);
    unreachable!("dot2_mod_p64 called outside CUDA kernel context")
}

/// Compute a three-term multiply-accumulate using the fixed 64-bit modulus.
#[inline(never)]
pub fn dot3_mod_p64(acc: u64, a0: u64, b0: u64, a1: u64, b1: u64, a2: u64, b2: u64) -> u64 {
    let _ = (acc, a0, b0, a1, b1, a2, b2);
    unreachable!("dot3_mod_p64 called outside CUDA kernel context")
}

/// Compute a four-term multiply-accumulate using the fixed 64-bit modulus.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn dot4_mod_p64(
    acc: u64,
    a0: u64,
    b0: u64,
    a1: u64,
    b1: u64,
    a2: u64,
    b2: u64,
    a3: u64,
    b3: u64,
) -> u64 {
    let _ = (acc, a0, b0, a1, b1, a2, b2, a3, b3);
    unreachable!("dot4_mod_p64 called outside CUDA kernel context")
}

/// Compute a five-term multiply-accumulate using the fixed 64-bit modulus.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn dot5_mod_p64(
    acc: u64,
    a0: u64,
    b0: u64,
    a1: u64,
    b1: u64,
    a2: u64,
    b2: u64,
    a3: u64,
    b3: u64,
    a4: u64,
    b4: u64,
) -> u64 {
    let _ = (acc, a0, b0, a1, b1, a2, b2, a3, b3, a4, b4);
    unreachable!("dot5_mod_p64 called outside CUDA kernel context")
}
