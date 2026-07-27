/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer carry-chain intrinsic translation.

use super::super::helpers::{emit_store_result_and_goto, insert_op};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{DotModP64Op, MulModP64PartialOp};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

#[allow(clippy::too_many_arguments)]
pub fn emit_mul_mod_p64_partial(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "mul_mod_p64_partial expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (a, after_a) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (b, after_b) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        after_a,
        loc.clone(),
    )?;
    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let op = Operation::new(
        ctx,
        MulModP64PartialOp::get_concrete_op_info(),
        vec![i64_type.to_ptr()],
        vec![a, b],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    insert_op(ctx, op, block_ptr, after_b);
    let result = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "mul_mod_p64_partial call without target block",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_dot_mod_p64(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !matches!(args.len(), 5 | 7 | 9 | 11) {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "dot_mod_p64 expects 5, 7, 9, or 11 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut operands = Vec::with_capacity(args.len());
    let mut after = prev_op;
    for arg in args {
        let (value, next) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, after, loc.clone())?;
        operands.push(value);
        after = next;
    }
    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let op = Operation::new(
        ctx,
        DotModP64Op::get_concrete_op_info(),
        vec![i64_type.to_ptr()],
        operands,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    insert_op(ctx, op, block_ptr, after);
    let result = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "dot_mod_p64 call without target block",
    )
}
