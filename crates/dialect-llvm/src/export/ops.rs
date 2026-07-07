/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Operation emission for LLVM IR.

use std::collections::HashMap;
use std::fmt::Write;

use pliron::r#type::Typed;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::{CallOpCallable, CallOpInterface},
    },
    context::Ptr,
    op::Op,
    operation::Operation,
    value::Value,
};

use crate::{
    attributes::{FCmpPredicateAttr, FPHalfAttr, GepIndexAttr, ICmpPredicateAttr},
    ops::{self, LlvmAtomicOpInterface},
    types::{FuncType, VoidType},
};

use super::{
    literals::{format_float_literal, format_half_literal},
    state::ModuleExportState,
};

/// Typed view of a dispatched LLVM dialect operation.
///
/// Each variant wraps a reference to the typed op so that handler methods can
/// call op-specific accessors (predicates, ordering, asm template, etc.)
/// without repeating the downcast inside every arm.
///
/// # Maintenance contract
///
/// Adding a new dialect-llvm op to textual export touches four places in
/// this file:
///
/// 1. Add a variant to `LlvmOp` below.
/// 2. Add a matching entry in the [`classify_op!`] invocation.
/// 3. Add an `emit_*` helper method on [`ModuleExportState`].
/// 4. Add a `Some(LlvmOp::X(op)) => self.emit_x(...)` arm in `export_op`.
///
/// Selective dispatch sites elsewhere in the export pipeline (e.g. the
/// value-name pre-pass in `function.rs`) inspect specific op types via
/// `downcast_ref` directly and do not need to be updated.
enum LlvmOp<'op> {
    // Terminators
    Return(&'op ops::ReturnOp),
    /// `UnreachableOp` carries no operands or attributes, so the typed reference
    /// is intentionally unread. The variant is kept tuple-shaped for uniformity
    /// with the other terminator variants.
    #[allow(dead_code)]
    Unreachable(&'op ops::UnreachableOp),
    Br(&'op ops::BrOp),
    CondBr(&'op ops::CondBrOp),
    // Memory
    Load(&'op ops::LoadOp),
    Store(&'op ops::StoreOp),
    Alloca(&'op ops::AllocaOp),
    GetElementPtr(&'op ops::GetElementPtrOp),
    // Atomics
    AtomicLoad(&'op ops::AtomicLoadOp),
    AtomicStore(&'op ops::AtomicStoreOp),
    AtomicRmw(&'op ops::AtomicRmwOp),
    AtomicCmpxchg(&'op ops::AtomicCmpxchgOp),
    Fence(&'op ops::FenceOp),
    // Integer arithmetic
    Add(&'op ops::AddOp),
    Sub(&'op ops::SubOp),
    Mul(&'op ops::MulOp),
    SDiv(&'op ops::SDivOp),
    UDiv(&'op ops::UDivOp),
    SRem(&'op ops::SRemOp),
    URem(&'op ops::URemOp),
    Shl(&'op ops::ShlOp),
    LShr(&'op ops::LShrOp),
    AShr(&'op ops::AShrOp),
    And(&'op ops::AndOp),
    Or(&'op ops::OrOp),
    Xor(&'op ops::XorOp),
    // Float arithmetic
    FAdd(&'op ops::FAddOp),
    FSub(&'op ops::FSubOp),
    FMul(&'op ops::FMulOp),
    FDiv(&'op ops::FDivOp),
    FRem(&'op ops::FRemOp),
    FNeg(&'op ops::FNegOp),
    // Comparison / select
    ICmp(&'op ops::ICmpOp),
    FCmp(&'op ops::FCmpOp),
    Select(&'op ops::SelectOp),
    // Calls and inline assembly
    Call(&'op ops::CallOp),
    InlineAsm(&'op ops::InlineAsmOp),
    InlineAsmMulti(&'op ops::InlineAsmMultiOp),
    // Casts
    Bitcast(&'op ops::BitcastOp),
    AddrSpaceCast(&'op ops::AddrSpaceCastOp),
    ZExt(&'op ops::ZExtOp),
    SExt(&'op ops::SExtOp),
    Trunc(&'op ops::TruncOp),
    PtrToInt(&'op ops::PtrToIntOp),
    IntToPtr(&'op ops::IntToPtrOp),
    UIToFP(&'op ops::UIToFPOp),
    SIToFP(&'op ops::SIToFPOp),
    FPToUI(&'op ops::FPToUIOp),
    FPToSI(&'op ops::FPToSIOp),
    FPExt(&'op ops::FPExtOp),
    FPTrunc(&'op ops::FPTruncOp),
    // Aggregates
    ExtractValue(&'op ops::ExtractValueOp),
    InsertValue(&'op ops::InsertValueOp),
    // Virtual / constant ops
    Undef(&'op ops::UndefOp),
    Constant(&'op ops::ConstantOp),
    AddressOf(&'op ops::AddressOfOp),
}

/// Try each `(Variant, OpType)` pair in order; return the first match.
///
/// Uses `return` to short-circuit out of the enclosing `try_from` body.
macro_rules! classify_op {
    ($op_obj:expr, { $($Variant:ident => $OpTy:ty),* $(,)? }) => {{
        let op = $op_obj;
        $(
            if let Some(inner) = op.downcast_ref::<$OpTy>() {
                return Ok(Self::$Variant(inner));
            }
        )*
        Err(())
    }};
}

impl<'op> TryFrom<&'op dyn Op> for LlvmOp<'op> {
    type Error = ();

    fn try_from(op_obj: &'op dyn Op) -> Result<Self, ()> {
        classify_op!(op_obj, {
            // Terminators
            Return       => ops::ReturnOp,
            Unreachable  => ops::UnreachableOp,
            Br           => ops::BrOp,
            CondBr       => ops::CondBrOp,
            // Memory
            Load         => ops::LoadOp,
            Store        => ops::StoreOp,
            Alloca       => ops::AllocaOp,
            GetElementPtr=> ops::GetElementPtrOp,
            // Atomics
            AtomicLoad   => ops::AtomicLoadOp,
            AtomicStore  => ops::AtomicStoreOp,
            AtomicRmw    => ops::AtomicRmwOp,
            AtomicCmpxchg=> ops::AtomicCmpxchgOp,
            Fence        => ops::FenceOp,
            // Integer arithmetic
            Add          => ops::AddOp,
            Sub          => ops::SubOp,
            Mul          => ops::MulOp,
            SDiv         => ops::SDivOp,
            UDiv         => ops::UDivOp,
            SRem         => ops::SRemOp,
            URem         => ops::URemOp,
            Shl          => ops::ShlOp,
            LShr         => ops::LShrOp,
            AShr         => ops::AShrOp,
            And          => ops::AndOp,
            Or           => ops::OrOp,
            Xor          => ops::XorOp,
            // Float arithmetic
            FAdd         => ops::FAddOp,
            FSub         => ops::FSubOp,
            FMul         => ops::FMulOp,
            FDiv         => ops::FDivOp,
            FRem         => ops::FRemOp,
            FNeg         => ops::FNegOp,
            // Comparison / select
            ICmp         => ops::ICmpOp,
            FCmp         => ops::FCmpOp,
            Select       => ops::SelectOp,
            // Calls and inline assembly
            Call         => ops::CallOp,
            InlineAsm    => ops::InlineAsmOp,
            InlineAsmMulti => ops::InlineAsmMultiOp,
            // Casts
            Bitcast      => ops::BitcastOp,
            AddrSpaceCast=> ops::AddrSpaceCastOp,
            ZExt         => ops::ZExtOp,
            SExt         => ops::SExtOp,
            Trunc        => ops::TruncOp,
            PtrToInt     => ops::PtrToIntOp,
            IntToPtr     => ops::IntToPtrOp,
            UIToFP       => ops::UIToFPOp,
            SIToFP       => ops::SIToFPOp,
            FPToUI       => ops::FPToUIOp,
            FPToSI       => ops::FPToSIOp,
            FPExt        => ops::FPExtOp,
            FPTrunc      => ops::FPTruncOp,
            // Aggregates
            ExtractValue => ops::ExtractValueOp,
            InsertValue  => ops::InsertValueOp,
            // Virtual / constant ops
            Undef        => ops::UndefOp,
            Constant     => ops::ConstantOp,
            AddressOf    => ops::AddressOfOp,
        })
    }
}

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_op(
        &mut self,
        op: Ptr<Operation>,
        value_names: &mut HashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let op_obj = Operation::get_op_dyn(op, self.ctx);

        // Register result names (skip if already named in pre-pass)
        for res in op_ref.results() {
            value_names.entry(res).or_insert_with(|| {
                let name = format!("%v{next_value_id}");
                *next_value_id += 1;
                name.clone()
            });
        }

        match LlvmOp::try_from(op_obj.as_ref()).ok() {
            // Terminators
            Some(LlvmOp::Return(op)) => self.emit_return(op, value_names, output)?,
            Some(LlvmOp::Unreachable(_)) => writeln!(output, "  unreachable").unwrap(),
            Some(LlvmOp::Br(op)) => self.emit_br(op, block_labels, output)?,
            Some(LlvmOp::CondBr(op)) => self.emit_cond_br(op, value_names, block_labels, output)?,
            // Memory
            Some(LlvmOp::Load(op)) => self.emit_load(op, value_names, output)?,
            Some(LlvmOp::Store(op)) => self.emit_store(op, value_names, output)?,
            Some(LlvmOp::Alloca(op)) => self.emit_alloca(op, value_names, output)?,
            Some(LlvmOp::GetElementPtr(op)) => self.emit_gep(op, value_names, output)?,
            // Atomics
            Some(LlvmOp::AtomicLoad(op)) => self.emit_atomic_load(op, value_names, output)?,
            Some(LlvmOp::AtomicStore(op)) => self.emit_atomic_store(op, value_names, output)?,
            Some(LlvmOp::AtomicRmw(op)) => self.emit_atomic_rmw(op, value_names, output)?,
            Some(LlvmOp::AtomicCmpxchg(op)) => self.emit_atomic_cmpxchg(op, value_names, output)?,
            Some(LlvmOp::Fence(op)) => self.emit_fence(op, output)?,
            // Integer arithmetic (all map to export_binop)
            Some(LlvmOp::Add(op)) => {
                self.export_binop("add", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Sub(op)) => {
                self.export_binop("sub", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Mul(op)) => {
                self.export_binop("mul", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SDiv(op)) => {
                self.export_binop("sdiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::UDiv(op)) => {
                self.export_binop("udiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SRem(op)) => {
                self.export_binop("srem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::URem(op)) => {
                self.export_binop("urem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Shl(op)) => {
                self.export_binop("shl", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::LShr(op)) => {
                self.export_binop("lshr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::AShr(op)) => {
                self.export_binop("ashr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::And(op)) => {
                self.export_binop("and", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Or(op)) => {
                self.export_binop("or", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Xor(op)) => {
                self.export_binop("xor", op.get_operation(), value_names, output)?
            }
            // Float arithmetic
            Some(LlvmOp::FAdd(op)) => {
                self.export_binop("fadd", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FSub(op)) => {
                self.export_binop("fsub", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FMul(op)) => {
                self.export_binop("fmul", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FDiv(op)) => {
                self.export_binop("fdiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FRem(op)) => {
                self.export_binop("frem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FNeg(op)) => self.emit_fneg(op, value_names, output)?,
            // Comparison / select
            Some(LlvmOp::ICmp(op)) => self.emit_icmp(op, value_names, output)?,
            Some(LlvmOp::FCmp(op)) => self.emit_fcmp(op, value_names, output)?,
            Some(LlvmOp::Select(op)) => self.emit_select(op, value_names, output)?,
            // Calls and inline assembly
            Some(LlvmOp::Call(op)) => self.emit_call(op, value_names, output)?,
            Some(LlvmOp::InlineAsm(op)) => self.emit_inline_asm(op, value_names, output)?,
            Some(LlvmOp::InlineAsmMulti(op)) => {
                self.emit_inline_asm_multi(op, value_names, output)?
            }
            // Casts
            Some(LlvmOp::Bitcast(op)) => {
                self.export_cast("bitcast", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::AddrSpaceCast(op)) => {
                self.export_cast("addrspacecast", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::ZExt(op)) => self.emit_zext(op, value_names, output)?,
            Some(LlvmOp::SExt(op)) => {
                self.export_cast("sext", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Trunc(op)) => {
                self.export_cast("trunc", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::PtrToInt(op)) => {
                self.export_cast("ptrtoint", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::IntToPtr(op)) => {
                self.export_cast("inttoptr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::UIToFP(op)) => {
                self.export_cast("uitofp", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SIToFP(op)) => {
                self.export_cast("sitofp", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPToUI(op)) => {
                self.export_cast("fptoui", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPToSI(op)) => {
                self.export_cast("fptosi", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPExt(op)) => {
                self.export_cast("fpext", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPTrunc(op)) => {
                self.export_cast("fptrunc", op.get_operation(), value_names, output)?
            }
            // Aggregates
            Some(LlvmOp::ExtractValue(op)) => self.emit_extract_value(op, value_names, output)?,
            Some(LlvmOp::InsertValue(op)) => self.emit_insert_value(op, value_names, output)?,
            // Virtual ops
            Some(LlvmOp::Undef(op)) => self.emit_undef(op, value_names),
            Some(LlvmOp::Constant(op)) => self.emit_constant(op, value_names),
            Some(LlvmOp::AddressOf(op)) => self.emit_address_of(op, value_names),
            // Unknown
            None => writeln!(
                output,
                "  ; Unknown op: {}",
                Operation::get_opid(op, self.ctx)
            )
            .unwrap(),
        }

        Ok(())
    }

    fn emit_return(
        &mut self,
        op: &ops::ReturnOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        write!(output, "  ret ").unwrap();
        if op_ref.operands().count() == 0 {
            write!(output, "void").unwrap();
        } else {
            let val = op_ref.operands().next().unwrap();
            self.export_type(val.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(val, value_names, output)?;
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_br(
        &self,
        op: &ops::BrOp,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let dest = op_ref.successors().next().unwrap();
        let label = block_labels.get(&dest).ok_or("Missing block label")?;
        writeln!(output, "  br label %{label}").unwrap();
        Ok(())
    }

    fn emit_cond_br(
        &mut self,
        op: &ops::CondBrOp,
        value_names: &HashMap<Value, String>,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let mut succs = op_ref.successors();
        let true_dest = succs.next().unwrap();
        let false_dest = succs.next().unwrap();
        let true_label = block_labels.get(&true_dest).ok_or("Missing true label")?;
        let false_label = block_labels.get(&false_dest).ok_or("Missing false label")?;
        let cond = op_ref.get_operand(0);

        write!(output, "  br i1 ").unwrap();
        self.export_value(cond, value_names, output)?;
        writeln!(output, ", label %{true_label}, label %{false_label}").unwrap();
        Ok(())
    }

    /// Render a type to a fresh `String` (typed-pointer helper).
    fn type_to_string(&self, ty: Ptr<pliron::r#type::TypeObj>) -> Result<String, String> {
        let mut s = String::new();
        self.export_type(ty, &mut s)?;
        Ok(s)
    }

    /// Typed-pointer form of a `<elem>*` pointer, preserving address space.
    fn typed_ptr_str(elem: &str, addrspace: u32) -> String {
        if addrspace != 0 {
            format!("{elem} addrspace({addrspace})*")
        } else {
            format!("{elem}*")
        }
    }

    /// Emit `  %tp.N = bitcast i8<as>* PTR to ELEM<as>*` and return the fresh temp name.
    /// The source is always `i8*` because every pointer SSA value is `i8*`-uniform in typed
    /// mode (globals are pre-registered as `i8*` bitcast constexprs, see `export_function`).
    fn tp_bitcast_ptr(
        &mut self,
        ptr_rendered: &str,
        addrspace: u32,
        elem: &str,
        output: &mut String,
    ) -> String {
        let tmp = self.fresh_tp_temp();
        let src = Self::typed_ptr_str("i8", addrspace);
        let dst = Self::typed_ptr_str(elem, addrspace);
        writeln!(output, "  {tmp} = bitcast {src} {ptr_rendered} to {dst}").unwrap();
        tmp
    }

    /// The pointee type a GEP produces: start at the source element type, then descend through
    /// each index AFTER the first (the first index is the base pointer stride and does not change
    /// the pointee). Array/vector indices descend to the element type; struct indices (always
    /// constant in LLVM) descend to the selected field.
    fn gep_result_elem_type(
        &self,
        src_elem: Ptr<pliron::r#type::TypeObj>,
        indices: &[GepIndexAttr],
    ) -> Ptr<pliron::r#type::TypeObj> {
        let mut cur = src_elem;
        for idx in indices.iter().skip(1) {
            let cur_ref = cur.deref(self.ctx);
            if let Some(arr) = cur_ref.downcast_ref::<crate::types::ArrayType>() {
                cur = arr.elem_type();
            } else if let Some(vec) = cur_ref.downcast_ref::<crate::types::VectorType>() {
                cur = vec.elem_type();
            } else if let Some(st) = cur_ref.downcast_ref::<crate::types::StructType>() {
                let field = match idx {
                    GepIndexAttr::Constant(k) => *k as usize,
                    GepIndexAttr::OperandIdx(_) => 0, // structs require constant indices
                };
                cur = st.fields().nth(field).unwrap_or(cur);
            } else {
                break;
            }
        }
        cur
    }

    /// Render a pointer operand to a `String`. In typed mode the operand's declared type is
    /// `i8<as>*`; this returns just the value token (e.g. `%v5` or an `@g` constexpr).
    fn ptr_operand_str(
        &self,
        ptr: Value,
        value_names: &HashMap<Value, String>,
    ) -> Result<String, String> {
        let mut s = String::new();
        self.export_value(ptr, value_names, &mut s)?;
        Ok(s)
    }

    fn emit_load(
        &mut self,
        op: &ops::LoadOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap().clone();
        let ty = res.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        if self.typed_pointers {
            let elem = self.type_to_string(ty)?;
            let ptr_str = self.ptr_operand_str(ptr, value_names)?;
            let typed_ptr = self.tp_bitcast_ptr(&ptr_str, addrspace, &elem, output);
            let dst = Self::typed_ptr_str(&elem, addrspace);
            writeln!(output, "  {res_name} = load {elem}, {dst} {typed_ptr}").unwrap();
            return Ok(());
        }

        write!(output, "  {res_name} = load ").unwrap();
        self.export_type(ty, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_store(
        &mut self,
        op: &ops::StoreOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        if self.typed_pointers {
            let elem = self.type_to_string(val.get_type(self.ctx))?;
            let val_str = {
                let mut s = String::new();
                self.export_value(val, value_names, &mut s)?;
                s
            };
            let ptr_str = self.ptr_operand_str(ptr, value_names)?;
            let typed_ptr = self.tp_bitcast_ptr(&ptr_str, addrspace, &elem, output);
            let dst = Self::typed_ptr_str(&elem, addrspace);
            writeln!(output, "  store {elem} {val_str}, {dst} {typed_ptr}").unwrap();
            return Ok(());
        }

        write!(output, "  store ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_alloca(
        &mut self,
        op: &ops::AllocaOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap().clone();
        let elem_ty = op
            .get_attr_alloca_element_type(self.ctx)
            .expect("Missing alloca_element_type")
            .get_type(self.ctx);

        if self.typed_pointers {
            // `alloca T` yields `T*`; uniformize to `i8*` for the canonical result name.
            let elem = self.type_to_string(elem_ty)?;
            let raw = format!("{res_name}.tpraw");
            writeln!(output, "  {raw} = alloca {elem}").unwrap();
            writeln!(output, "  {res_name} = bitcast {elem}* {raw} to i8*").unwrap();
            return Ok(());
        }

        write!(output, "  {res_name} = alloca ").unwrap();
        self.export_type(elem_ty, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_gep(
        &mut self,
        op: &ops::GetElementPtrOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap().clone();
        let ptr = op_ref.get_operand(0);
        let elem_ty = op
            .get_attr_gep_src_elem_type(self.ctx)
            .expect("Missing gep_src_elem_type")
            .get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        // Render the index list once (shared by both modes).
        let mut indices_str = String::new();
        for idx_attr in &op.get_attr_gep_indices(self.ctx).unwrap().0 {
            write!(indices_str, ", ").unwrap();
            match idx_attr {
                GepIndexAttr::Constant(val) => {
                    write!(indices_str, "i32 {val}").unwrap();
                }
                GepIndexAttr::OperandIdx(operand_idx) => {
                    let val = op_ref.get_operand(*operand_idx);
                    self.export_type(val.get_type(self.ctx), &mut indices_str)?;
                    write!(indices_str, " ").unwrap();
                    self.export_value(val, value_names, &mut indices_str)?;
                }
            }
        }

        if self.typed_pointers {
            let src_elem = self.type_to_string(elem_ty)?;
            let ptr_str = self.ptr_operand_str(ptr, value_names)?;
            let typed_base = self.tp_bitcast_ptr(&ptr_str, addrspace, &src_elem, output);
            let src_ptr = Self::typed_ptr_str(&src_elem, addrspace);
            let raw = format!("{res_name}.tpraw");
            writeln!(
                output,
                "  {raw} = getelementptr inbounds {src_elem}, {src_ptr} {typed_base}{indices_str}"
            )
            .unwrap();
            // Uniformize the result (a `U addrspace(A)*`) back to `i8 addrspace(A)*`.
            let result_elem_ty =
                self.gep_result_elem_type(elem_ty, &op.get_attr_gep_indices(self.ctx).unwrap().0);
            let result_elem = self.type_to_string(result_elem_ty)?;
            let result_ptr = Self::typed_ptr_str(&result_elem, addrspace);
            let dst = Self::typed_ptr_str("i8", addrspace);
            writeln!(output, "  {res_name} = bitcast {result_ptr} {raw} to {dst}").unwrap();
            return Ok(());
        }

        write!(output, "  {res_name} = getelementptr inbounds ").unwrap();
        self.export_type(elem_ty, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        write!(output, "{indices_str}").unwrap();
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_atomic_load(
        &mut self,
        op: &ops::AtomicLoadOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();
        let ty = res.get_type(self.ctx);
        let syncscope = ops::atomic::format_syncscope(&op.syncscope(self.ctx));
        let ordering = ops::atomic::format_ordering(&op.ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  {res_name} = load atomic ").unwrap();
        self.export_type(ty, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        let align = self.natural_alignment(ty);
        writeln!(output, "{syncscope} {ordering}, align {align}").unwrap();
        Ok(())
    }

    fn emit_atomic_store(
        &mut self,
        op: &ops::AtomicStoreOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);
        let syncscope = ops::atomic::format_syncscope(&op.syncscope(self.ctx));
        let ordering = ops::atomic::format_ordering(&op.ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  store atomic ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        let align = self.natural_alignment(val.get_type(self.ctx));
        writeln!(output, "{syncscope} {ordering}, align {align}").unwrap();
        Ok(())
    }

    fn emit_atomic_rmw(
        &mut self,
        op: &ops::AtomicRmwOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();
        let rmw_kind = ops::atomic::format_rmw_kind(&op.rmw_kind(self.ctx));
        let syncscope = ops::atomic::format_syncscope(&op.syncscope(self.ctx));
        let ordering = ops::atomic::format_ordering(&op.ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  {res_name} = atomicrmw {rmw_kind} ").unwrap();
        write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        writeln!(output, "{syncscope} {ordering}").unwrap();
        Ok(())
    }

    fn emit_atomic_cmpxchg(
        &mut self,
        op: &ops::AtomicCmpxchgOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let cmp = op_ref.get_operand(1);
        let new_val = op_ref.get_operand(2);
        let res_name = value_names.get(&res).unwrap();
        let success_ord = ops::atomic::format_ordering(&op.success_ordering(self.ctx));
        let failure_ord = ops::atomic::format_ordering(&op.failure_ordering(self.ctx));
        let syncscope = ops::atomic::format_syncscope(&op.syncscope(self.ctx));
        let val_ty = cmp.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        // Emit cmpxchg returning { T, i1 }
        let struct_name = format!("{res_name}.cx");
        write!(output, "  {struct_name} = cmpxchg ").unwrap();
        write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(cmp, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(new_val, value_names, output)?;
        writeln!(output, "{syncscope} {success_ord} {failure_ord}").unwrap();

        // Extract old value (element 0 of { T, i1 })
        write!(output, "  {res_name} = extractvalue {{ ").unwrap();
        self.export_type(val_ty, output)?;
        writeln!(output, ", i1 }} {struct_name}, 0").unwrap();
        Ok(())
    }

    fn emit_fence(&self, op: &ops::FenceOp, output: &mut String) -> Result<(), String> {
        let syncscope = ops::atomic::format_syncscope(&op.syncscope(self.ctx));
        let ordering = ops::atomic::format_ordering(&op.ordering(self.ctx));
        writeln!(output, "  fence{syncscope} {ordering}").unwrap();
        Ok(())
    }

    fn emit_fneg(
        &mut self,
        op: &ops::FNegOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let arg = op_ref.get_operand(0);

        write!(output, "  {res_name} = fneg ").unwrap();
        self.export_type(arg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(arg, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_icmp(
        &mut self,
        op: &ops::ICmpOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred = match op.predicate(self.ctx) {
            ICmpPredicateAttr::EQ => "eq",
            ICmpPredicateAttr::NE => "ne",
            ICmpPredicateAttr::SLT => "slt",
            ICmpPredicateAttr::SLE => "sle",
            ICmpPredicateAttr::SGT => "sgt",
            ICmpPredicateAttr::SGE => "sge",
            ICmpPredicateAttr::ULT => "ult",
            ICmpPredicateAttr::ULE => "ule",
            ICmpPredicateAttr::UGT => "ugt",
            ICmpPredicateAttr::UGE => "uge",
        };

        write!(output, "  {res_name} = icmp {pred} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_fcmp(
        &mut self,
        op: &ops::FCmpOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred = match op.predicate(self.ctx) {
            FCmpPredicateAttr::False => "false",
            FCmpPredicateAttr::OEQ => "oeq",
            FCmpPredicateAttr::OGT => "ogt",
            FCmpPredicateAttr::OGE => "oge",
            FCmpPredicateAttr::OLT => "olt",
            FCmpPredicateAttr::OLE => "ole",
            FCmpPredicateAttr::ONE => "one",
            FCmpPredicateAttr::ORD => "ord",
            FCmpPredicateAttr::UEQ => "ueq",
            FCmpPredicateAttr::UGT => "ugt",
            FCmpPredicateAttr::UGE => "uge",
            FCmpPredicateAttr::ULT => "ult",
            FCmpPredicateAttr::ULE => "ule",
            FCmpPredicateAttr::UNE => "une",
            FCmpPredicateAttr::UNO => "uno",
            FCmpPredicateAttr::True => "true",
        };

        write!(output, "  {res_name} = fcmp {pred} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_select(
        &mut self,
        op: &ops::SelectOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let cond = op_ref.get_operand(0);
        let true_val = op_ref.get_operand(1);
        let false_val = op_ref.get_operand(2);
        let val_ty = true_val.get_type(self.ctx);

        write!(output, "  {res_name} = select i1 ").unwrap();
        self.export_value(cond, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(true_val, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(false_val, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_call(
        &mut self,
        op: &ops::CallOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let callee = op.callee(self.ctx);
        let func_ty = op.callee_type(self.ctx);
        let func_ty_ref = func_ty.deref(self.ctx);
        let llvm_func_ty = func_ty_ref.downcast_ref::<FuncType>().unwrap();
        let ret_ty = llvm_func_ty.result_type();
        let is_void = ret_ty.deref(self.ctx).is::<VoidType>();

        // Void calls: "call void @func(...)"
        // Non-void:   "%vN = call <type> @func(...)"
        if is_void {
            write!(output, "  call void").unwrap();
        } else {
            let res = op_ref.get_result(0);
            let res_name = value_names.get(&res).unwrap();
            write!(output, "  {res_name} = call ").unwrap();
            self.export_type(ret_ty, output)?;
        }

        let mut is_convergent = false;
        match callee {
            CallOpCallable::Direct(identifier) => {
                let name = identifier.to_string();
                // LLVM intrinsics use dots in IR; Pliron IR identifiers use underscores.
                let fixed = if name.starts_with("llvm_") {
                    name.replace('_', ".")
                } else {
                    super::names::strip_device_prefix(&name)
                };
                is_convergent = Self::is_convergent_intrinsic(&fixed);
                write!(output, " @{fixed}(").unwrap();
            }
            CallOpCallable::Indirect(val) => {
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output).unwrap();
                write!(output, "(").unwrap();
            }
        }

        for (i, arg) in op_ref.operands().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(arg.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(arg, value_names, output)?;
        }

        if is_convergent {
            writeln!(output, ") #0").unwrap();
            self.convergent_used = true;
        } else {
            writeln!(output, ")").unwrap();
        }
        Ok(())
    }

    fn emit_inline_asm(
        &mut self,
        op: &ops::InlineAsmOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let asm_template = op.asm_template(self.ctx);
        let constraints = op.constraints(self.ctx);
        let is_convergent = op.is_convergent(self.ctx);

        if op_ref.get_num_results() > 0 {
            let res = op_ref.get_result(0);
            let res_name = value_names.get(&res).unwrap();
            let res_ty = res.get_type(self.ctx);
            write!(output, "  {res_name} = call ").unwrap();
            self.export_type(res_ty, output)?;
        } else {
            write!(output, "  call void").unwrap();
        }

        write!(
            output,
            " asm sideeffect \"{asm_template}\", \"{constraints}\"("
        )
        .unwrap();
        for (i, arg) in op_ref.operands().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(arg.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(arg, value_names, output)?;
        }

        if is_convergent {
            writeln!(output, ") #0").unwrap();
            self.convergent_used = true;
        } else {
            writeln!(output, ")").unwrap();
        }
        Ok(())
    }

    fn emit_inline_asm_multi(
        &mut self,
        op: &ops::InlineAsmMultiOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let asm_template = op.asm_template(self.ctx);
        let constraints = op.constraints(self.ctx);
        let is_convergent = op.is_convergent(self.ctx);
        let num_results = op_ref.get_num_results();

        if num_results == 0 {
            write!(output, "  call void").unwrap();
            write!(
                output,
                " asm sideeffect \"{asm_template}\", \"{constraints}\"("
            )
            .unwrap();
            for (i, arg) in op_ref.operands().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(arg, value_names, output)?;
            }
            if is_convergent {
                writeln!(output, ") #0").unwrap();
                self.convergent_used = true;
            } else {
                writeln!(output, ")").unwrap();
            }
        } else {
            // Build the struct return type
            let mut struct_type = String::from("{");
            for i in 0..num_results {
                if i > 0 {
                    struct_type.push_str(", ");
                }
                let res_ty = op_ref.get_result(i).get_type(self.ctx);
                let mut ty_str = String::new();
                self.export_type(res_ty, &mut ty_str)?;
                struct_type.push_str(&ty_str);
            }
            struct_type.push('}');

            let first_res_name = value_names.get(&op_ref.get_result(0)).unwrap();
            let struct_result_name = format!("{first_res_name}_struct");

            write!(output, "  {struct_result_name} = call {struct_type}").unwrap();
            write!(
                output,
                " asm sideeffect \"{asm_template}\", \"{constraints}\"("
            )
            .unwrap();
            for (i, arg) in op_ref.operands().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(arg, value_names, output)?;
            }
            if is_convergent {
                writeln!(output, ") #0").unwrap();
                self.convergent_used = true;
            } else {
                writeln!(output, ")").unwrap();
            }

            for i in 0..num_results {
                let res_name = value_names.get(&op_ref.get_result(i)).unwrap();
                writeln!(
                    output,
                    "  {res_name} = extractvalue {struct_type} {struct_result_name}, {i}"
                )
                .unwrap();
            }
        }
        Ok(())
    }

    fn emit_zext(
        &mut self,
        op: &ops::ZExtOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let val = op_ref.get_operand(0);

        // Manual attribute access since helper is missing
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        let nneg = op_ref
            .attributes
            .0
            .get(&nneg_key)
            .and_then(|attr| {
                attr.downcast_ref::<pliron::builtin::attributes::BoolAttr>()
                    .map(|b| bool::from(b.clone()))
            })
            .unwrap_or(false);

        write!(output, "  {res_name} = zext ").unwrap();
        if nneg {
            write!(output, "nneg ").unwrap();
        }
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_extract_value(
        &mut self,
        op: &ops::ExtractValueOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);

        write!(output, "  {res_name} = extractvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        for idx in op.indices(self.ctx) {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_insert_value(
        &mut self,
        op: &ops::InsertValueOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);

        write!(output, "  {res_name} = insertvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        for idx in op.indices(self.ctx) {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_undef(&self, op: &ops::UndefOp, value_names: &mut HashMap<Value, String>) {
        let res = op.get_operation().deref(self.ctx).get_result(0);
        value_names.insert(res, "undef".to_string());
    }

    fn emit_constant(&self, op: &ops::ConstantOp, value_names: &mut HashMap<Value, String>) {
        let val_attr = op.get_value(self.ctx);
        let const_str = if let Some(int_attr) = val_attr.downcast_ref::<IntegerAttr>() {
            // Use APInt's proper decimal string conversion instead of parsing debug format.
            // The old code parsed debug strings like "APInt { value: 0x4000_0000_0000_u64 }"
            // by splitting on '_', which broke for values with underscore grouping
            // (e.g., 1u64 << 46 = 0x4000_0000_0000 would become 0x4000 = 16384).
            int_attr.value().to_string_unsigned_decimal()
        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
            format_half_literal(fp16_attr.to_bits())
        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
            let float_val: f32 = fp32_attr.clone().into();
            format_float_literal(f64::from(float_val))
        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
            let float_val: f64 = fp64_attr.clone().into();
            format_float_literal(float_val)
        } else {
            "0".to_string() // Fallback
        };

        let res = op.get_operation().deref(self.ctx).get_result(0);
        value_names.insert(res, const_str);
    }

    fn emit_address_of(&self, op: &ops::AddressOfOp, value_names: &HashMap<Value, String>) {
        // AddressOfOp is virtual in textual LLVM IR: every use site prints the
        // global symbol directly. The naming pre-pass in export_function
        // registers the result as `@<global_name>` before any block is emitted,
        // so there is nothing to emit here. The assertion keeps the contract
        // honest if the pre-pass is ever refactored.
        let res = op.get_operation().deref(self.ctx).get_result(0);
        // Opaque mode registers the result as `@name`; typed-pointer mode registers it as an
        // `i8*` bitcast constexpr (`bitcast (<sig>* @name to i8*)`) so the global is pointer-
        // uniform. Both are virtual — every use site prints the registered token inline.
        debug_assert!(
            value_names.get(&res).is_some_and(|name| name.starts_with('@')
                || name.starts_with("bitcast (")),
            "AddressOfOp result must be pre-registered as a global symbol (or typed-pointer \
             bitcast constexpr) by the naming pre-pass; got {:?}",
            value_names.get(&res),
        );
    }

    pub(super) fn export_binop(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    /// Export a cast: `%res = <op_name> <src_type> <val> to <dst_type>`
    pub(super) fn export_cast(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let val = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_value(
        &self,
        val: Value,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        if let Some(name) = value_names.get(&val) {
            write!(output, "{name}").unwrap();
        } else {
            write!(output, "undef").unwrap();
        }
        Ok(())
    }
}

/// Return the address space of a pointer type, or 0 for non-pointer types.
fn addrspace_of(ty: Ptr<pliron::r#type::TypeObj>, ctx: &pliron::context::Context) -> u32 {
    ty.deref(ctx)
        .downcast_ref::<crate::types::PointerType>()
        .map_or(0, crate::types::PointerType::address_space)
}

/// Format the pointer operand prefix for memory instructions.
///
/// Returns `"ptr addrspace(N) "` for non-default address spaces, `"ptr "` otherwise.
fn ptr_qualifier(addrspace: u32) -> String {
    if addrspace != 0 {
        format!("ptr addrspace({addrspace}) ")
    } else {
        "ptr ".to_string()
    }
}
