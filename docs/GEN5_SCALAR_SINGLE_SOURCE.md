# Gen5 scalar single-source operations

The compile-time scalar evaluator must keep single-source SOP1 operations on
their one-operand path. Previously `SAbsI32`, `SFlbitI32B32`, `SFF1I32B64`,
and `SBcnt1I32B64` fell through to the two-source path and failed shader
translation with a missing `scalar-source1`.

The evaluator now mirrors the runtime SPIR-V semantics: signed absolute value
uses unsigned two's-complement arithmetic (so `INT_MIN` remains representable),
bit scans return the documented all-ones sentinel for zero, and the 64-bit
forms read a scalar register pair while producing a 32-bit result.

This is the focused scalar portion of upstream
[#745](https://github.com/sharpemu/sharpemu/pull/745); larger stale-base shader,
GUI, and presenter changes from that PR are intentionally not imported here.
