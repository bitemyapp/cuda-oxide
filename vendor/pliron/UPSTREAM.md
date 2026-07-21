# Vendored pliron

Source: https://github.com/vaivaswatha/pliron

Revision: `b51e73b11648508188184451adebdcf63957b7fe`

The snapshot is vendored so cuda-oxide can carry a narrowly scoped mem2reg
complexity fix while preserving a reproducible compiler dependency. The local
change is documented beside the indexed rename walk in `src/opts/mem2reg.rs`.
