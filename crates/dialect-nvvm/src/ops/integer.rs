/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Multi-instruction integer carry-chain operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Fixed-modulus 64-bit multiply with a partially reduced result.
#[pliron_op(
    name = "nvvm.mul_mod_p64_partial",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MulModP64PartialOp;

impl MulModP64PartialOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Fixed-modulus dot product with two to five product terms and one canonical result.
///
/// The importer validates the dynamic operand count (`acc` plus two operands per term).
#[pliron_op(
    name = "nvvm.dot_mod_p64",
    format,
    verifier = "succ",
    interfaces = [NResultsInterface<1>],
)]
pub struct DotModP64Op;

impl DotModP64Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    MulModP64PartialOp::register(ctx);
    DotModP64Op::register(ctx);
}
