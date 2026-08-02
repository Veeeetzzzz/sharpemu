# SharpEmu v76 source checkpoint

This checkpoint completes the documented legacy single-precision multiply
family for the current RDNA2/GFX10 ISA pass. In addition to the VOP2/VOP3
`V_MUL_LEGACY_F32` and `V_MAC_LEGACY_F32` paths, Vulkan/SPIR-V and Metal/MSL
now translate the explicit-addend `V_MAD_LEGACY_F32` companion at the
documented RDNA2 opcode 320 (`0x140`). The backend preserves the DX9-style
zero-product behavior used by the legacy operations. Exact LLVM GFX10 decode
fixtures and focused Vulkan/Metal translation tests accompany the change.

Validation completed on 2026-08-02:

- `git diff --check`
- `dotnet test SharpEmu.slnx --no-restore -v:minimal`
  - ShaderCompiler: 269 passed
  - ShaderCompiler.Metal: 210 passed
  - Libs: 850 passed
  - Total: 1,329 passed, 0 failed
- Release CLI publish completed successfully for `win-x64` with self-contained output
- NU1900 vulnerability-feed warning is environmental (NuGet service unavailable)

Runnable Windows checkpoint:

`C:\Users\W10\Desktop\sharpemu-win64-29021b5\sharpemu-v76-legacy-mad-checkpoint-cli\SharpEmu.exe`

SHA-256:

`A680BCCE2C0844BA154282938DE255E87966EAB9E8D7CC0FB7F65BE3B22BCA0A`

The repository's `.git` directory is read-only in this workspace, so the
checkpoint is recorded as this source marker and runnable publish folder
rather than a local Git commit.
