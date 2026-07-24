/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-uniform global-memory operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Load a `u64` from a read-only global address that is identical in every active warp lane.
///
/// PTX: `ldu.global.u64 dst, [address];`
#[pliron_op(
    name = "nvvm.ldu_global_u64",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct LduGlobalU64Op;

impl LduGlobalU64Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

pub fn register(ctx: &mut Context) {
    LduGlobalU64Op::register(ctx);
}
