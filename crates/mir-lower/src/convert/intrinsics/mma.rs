/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix multiply-accumulate conversion.

use dialect_llvm::{ops as llvm, types as llvm_types};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        inserter::Inserter,
        rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    result::Result,
};

pub(crate) fn convert_mma_m16n8k32_u8(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty.to_ptr(); 4]);
    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        operands,
        concat!(
            "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 ",
            "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
        ),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4_u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![index])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        results.push(extract_op.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}
