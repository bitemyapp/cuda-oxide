/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix multiply-accumulate operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32`.
#[pliron_op(
    name = "nvvm.mma_m16n8k32_u8",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<10>, NResultsInterface<4>],
)]
pub struct MmaM16n8k32U8Op;

impl MmaM16n8k32U8Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    MmaM16n8k32U8Op::register(ctx);
}
