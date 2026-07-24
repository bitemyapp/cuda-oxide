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
