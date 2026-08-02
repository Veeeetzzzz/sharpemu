# SharpEmu v77 source checkpoint

This checkpoint adds the documented RDNA2/GFX10 `V_MULLIT_F32` VOP3 family
(opcode 336 decimal, `0x150`). The decoder accepts the exact LLVM GFX10
three-source encoding, while Vulkan/SPIR-V and Metal/MSL emit the documented
`S0*S1` operation with the specified `0.0*x = 0.0` rule. No undocumented
NaN, infinity, or overflow behavior is synthesized; those values remain under
the target floating-point multiply semantics.

Validation completed on 2026-08-02:

- `git diff --check`
- Focused coverage: 5 ShaderCompiler tests and 4 ShaderCompiler.Metal tests
- `dotnet test SharpEmu.slnx --no-restore -v:minimal`
  - ShaderCompiler: 272 passed
  - ShaderCompiler.Metal: 211 passed
  - Libs: 850 passed
  - Total: 1,333 passed, 0 failed
- Release CLI publish completed successfully for `win-x64` with self-contained output
- NU1900 vulnerability-feed warning is environmental (NuGet service unavailable)

Runnable Windows checkpoint:

`C:\Users\W10\Desktop\sharpemu-win64-29021b5\sharpemu-v77-mullit-f32-checkpoint-cli\SharpEmu.exe`

SHA-256:

`1F891C5DB622D1D8DFBAA306D1EAA81D3715AAE105D5B7BB539D12BC01E4E12A`

The repository's `.git` directory is read-only in this workspace, so the
checkpoint is recorded as this source marker and runnable publish folder
rather than a local Git commit.
