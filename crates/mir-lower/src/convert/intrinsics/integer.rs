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
use std::fmt::Write;

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

pub(crate) fn convert_dot_mod_p64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    // Five-limb accumulation/reduction for full-width residues; only the reduction is amortized.
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if !matches!(operands.len(), 5 | 7 | 9 | 11) {
        return pliron::input_err_noloc!("dot_mod_p64 requires 5, 7, 9, or 11 operands");
    }
    let terms = (operands.len() - 1) / 2;
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let mut ptx = String::from(
        "{ .reg .u32 %a0, %a1, %b0, %b1; \
         .reg .u32 %e0, %e1, %e2, %e3, %e4; \
         .reg .u32 %o0, %o1, %o2, %carry; \
         .reg .u64 %value, %reduced; .reg .pred %top; ",
    );

    // The first product initializes the disjoint even and cross-term accumulators.
    ptx.push_str(
        "mov.b64 {%a0, %a1}, $2; mov.b64 {%b0, %b1}, $3; \
         mul.lo.u32 %e0, %a0, %b0; mul.hi.u32 %e1, %a0, %b0; \
         mul.lo.u32 %e2, %a1, %b1; mul.hi.u32 %e3, %a1, %b1; mov.u32 %e4, 0; \
         mul.lo.u32 %o0, %a0, %b1; mul.hi.u32 %o1, %a0, %b1; \
         mad.lo.cc.u32 %o0, %a1, %b0, %o0; \
         madc.hi.cc.u32 %o1, %a1, %b0, %o1; addc.u32 %o2, 0, 0; ",
    );
    for term in 1..terms {
        let a_index = 2 + term * 2;
        let b_index = a_index + 1;
        write!(
            ptx,
            "mov.b64 {{%a0, %a1}}, ${a_index}; mov.b64 {{%b0, %b1}}, ${b_index}; \
             mad.lo.cc.u32 %e0, %a0, %b0, %e0; \
             madc.hi.cc.u32 %e1, %a0, %b0, %e1; \
             madc.lo.cc.u32 %e2, %a1, %b1, %e2; \
             madc.hi.cc.u32 %e3, %a1, %b1, %e3; addc.u32 %e4, %e4, 0; \
             mad.lo.cc.u32 %o0, %a1, %b0, %o0; \
             madc.hi.cc.u32 %o1, %a1, %b0, %o1; addc.u32 %o2, %o2, 0; \
             mad.lo.cc.u32 %o0, %a0, %b1, %o0; \
             madc.hi.cc.u32 %o1, %a0, %b1, %o1; addc.u32 %o2, %o2, 0; "
        )
        .expect("writing PTX string cannot fail");
    }

    ptx.push_str(
        "add.cc.u32 %e1, %e1, %o0; addc.cc.u32 %e2, %e2, %o1; \
         addc.cc.u32 %e3, %e3, %o2; addc.u32 %e4, %e4, 0; \
         mov.b64 {%a0, %a1}, $1; \
         add.cc.u32 %e0, %e0, %a0; addc.cc.u32 %e1, %e1, %a1; \
         addc.cc.u32 %e2, %e2, 0; addc.cc.u32 %e3, %e3, 0; addc.u32 %e4, %e4, 0; \
         sub.cc.u32 %e1, %e1, %e4; subc.cc.u32 %e2, %e2, 0; \
         subc.cc.u32 %e3, %e3, 0; subc.u32 %e4, %e4, %e4; sub.u32 %e2, %e2, %e4; \
         mad.lo.cc.u32 %e0, %e2, 0xffffffff, %e0; \
         madc.hi.cc.u32 %e1, %e2, 0xffffffff, %e1; addc.u32 %carry, 0, 0; \
         sub.cc.u32 %e0, %e0, %e3; subc.cc.u32 %e1, %e1, 0; \
         subc.u32 %carry, %carry, 0; \
         neg.s32 %e2, %carry; shr.s32 %e2, %e2, 1; \
         sub.cc.u32 %e0, %e0, %carry; subc.u32 %e1, %e1, %e2; \
         mov.b64 %value, {%e0, %e1}; \
         add.cc.u64 %reduced, %value, 0xffffffff; addc.u32 %carry, 0, 0; \
         setp.ne.u32 %top, %carry, 0; selp.u64 $0, %reduced, %value, %top; }",
    );
    let mut constraints = String::from("=l");
    for _ in &operands {
        constraints.push_str(",l");
    }
    let inline_asm = llvm::InlineAsmOp::new(ctx, i64_ty.into(), operands, &ptx, &constraints);
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
