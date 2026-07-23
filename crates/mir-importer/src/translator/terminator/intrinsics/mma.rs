/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix multiply-accumulate intrinsics.

use super::super::helpers::emit_store_result_and_goto;
use super::tcgen05::destination_struct_type;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::{rvalue, values::ValueMap};
use dialect_nvvm::ops::MmaM16n8k32U8Op;
use pliron::{
    basic_block::BasicBlock,
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    input_err,
    location::{Located, Location},
    op::Op,
    operation::Operation,
    value::Value,
};
use rustc_public::mir;

#[allow(clippy::too_many_arguments)]
pub fn emit_mma_m16n8k32_u8(
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
    if args.len() != 10 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_m16n8k32_u8 expects 10 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(10);
    for arg in args {
        let (value, next_op) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        operands.push(value);
        last_op = next_op;
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let mma_op = Operation::new(
        ctx,
        MmaM16n8k32U8Op::get_concrete_op_info(),
        vec![i32_ty.to_ptr(); 4],
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    let results: Vec<Value> = (0..4).map(|i| mma_op.deref(ctx).get_result(i)).collect();
    let array_ty = dialect_mir::types::MirArrayType::get(ctx, i32_ty.to_ptr(), 4);
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, mma_op);

    let struct_ty = destination_struct_type(ctx, body, destination, loc.clone())?;
    let array_result = array_op.deref(ctx).get_result(0);
    let struct_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructStructOp::get_concrete_op_info(),
        vec![struct_ty],
        vec![array_result],
        vec![],
        0,
    );
    struct_op.deref_mut(ctx).set_loc(loc.clone());
    struct_op.insert_after(ctx, array_op);

    let struct_result = struct_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        struct_result,
        target,
        block_ptr,
        struct_op,
        value_map,
        block_map,
        loc,
        "mma_m16n8k32_u8 call without target block",
    )
}
