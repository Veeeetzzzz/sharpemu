# SharpEmu v74 source checkpoint

This checkpoint extends the RDNA2/GFX10 ISA pass with `V_DIV_FMAS_F32`
(opcode 367 / GFX10 opcode `0x16F`). The decoder now recognizes the exact
LLVM GFX10 encoding, and both Vulkan/SPIR-V and Metal/MSL translate the
documented VCC-conditioned fused multiply-add scaling by `2^32`. Focused
decode and backend translation tests accompany the implementation.

Validation completed on 2026-08-02:

- `git diff --check`
- `dotnet test SharpEmu.slnx --no-restore -v:minimal`
  - ShaderCompiler: 261 passed
  - ShaderCompiler.Metal: 207 passed
  - Libs: 850 passed
- Release CLI publish completed successfully for `win-x64` and self-contained output
- NU1900 vulnerability-feed warning is environmental (NuGet service unavailable)

Runnable Windows checkpoint:

`C:\Users\W10\Desktop\sharpemu-win64-29021b5\sharpemu-v74-div-fmas-checkpoint-cli\SharpEmu.exe`

SHA-256:

`F0BF5832B4C21041702D8B3AA01E216F76A8303A887C06989433609B836D647D`

The repository's `.git` directory is read-only in this workspace, so the
checkpoint is recorded as this source marker and runnable publish folder
rather than a local Git commit.
