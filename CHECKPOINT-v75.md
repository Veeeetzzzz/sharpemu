# SharpEmu v75 source checkpoint

This checkpoint extends the RDNA2/GFX10 ISA pass with the documented legacy
single-precision multiply family: `V_FMAC_LEGACY_F32` and
`V_MUL_LEGACY_F32` in both VOP2 and VOP3 encodings. Vulkan/SPIR-V and
Metal/MSL now enforce the DX9 rule that a zero operand produces a zero product,
including the legacy multiply-accumulate form. Exact LLVM GFX10 decode fixtures
and backend translation tests accompany the implementation.

Validation completed on 2026-08-02:

- `git diff --check`
- `dotnet test SharpEmu.slnx --no-restore -v:minimal`
  - ShaderCompiler: 267 passed
  - ShaderCompiler.Metal: 209 passed
  - Libs: 850 passed
- Release CLI publish completed successfully for `win-x64` and self-contained output
- NU1900 vulnerability-feed warning is environmental (NuGet service unavailable)

Runnable Windows checkpoint:

`C:\Users\W10\Desktop\sharpemu-win64-29021b5\sharpemu-v75-legacy-float-checkpoint-cli\SharpEmu.exe`

SHA-256:

`BC4338C5B564E5D6F10118C53185DEAD6FC84FD1E78B7C55A8C03F0C35DFC5EA`

The repository's `.git` directory is read-only in this workspace, so the
checkpoint is recorded as this source marker and runnable publish folder
rather than a local Git commit.
