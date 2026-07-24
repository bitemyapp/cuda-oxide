/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer carry-chain intrinsic lowering.

use dialect_llvm::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_mul_mod_p64_partial(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mul_mod_p64_partial requires 2 operands");
    }
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let ptx = concat!(
        "{ ",
        ".reg .u32 %a0, %a1, %b0, %b1; ",
        ".reg .u32 %t0, %t1, %t2, %t3, %carry; ",
        "mov.b64 {%a0, %a1}, $1; mov.b64 {%b0, %b1}, $2; ",
        "mul.lo.u32 %t0, %a0, %b0; mul.hi.u32 %t1, %a0, %b0; ",
        "mul.lo.u32 %t2, %a1, %b1; mul.hi.u32 %t3, %a1, %b1; ",
        "mad.lo.cc.u32 %t1, %a0, %b1, %t1; ",
        "madc.hi.cc.u32 %t2, %a0, %b1, %t2; addc.u32 %carry, 0, 0; ",
        "mad.lo.cc.u32 %t1, %a1, %b0, %t1; ",
        "madc.hi.cc.u32 %t2, %a1, %b0, %t2; addc.u32 %t3, %t3, %carry; ",
        "mad.lo.cc.u32 %t0, %t2, 0xffffffff, %t0; ",
        "madc.hi.cc.u32 %t1, %t2, 0xffffffff, %t1; addc.u32 %t2, 0, 0; ",
        "sub.cc.u32 %t0, %t0, %t3; subc.cc.u32 %t1, %t1, 0; ",
        "subc.u32 %t2, %t2, 0; ",
        "neg.s32 %carry, %t2; shr.s32 %carry, %carry, 1; ",
        "sub.cc.u32 %t0, %t0, %t2; subc.u32 %t1, %t1, %carry; ",
        "mov.b64 $0, {%t0, %t1}; ",
        "}"
    );
    let inline_asm = llvm::InlineAsmOp::new(ctx, i64_ty.into(), operands, ptx, "=l,l,l");
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
