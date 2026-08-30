# SharpEmu v72 source checkpoint

This checkpoint packages the current source worktree after the RDNA2/GFX10
ISA coverage pass and the earlier Vulkan/Metal video lifetime and detile
fixes. The shader pass includes 64-bit vector compares and shifts, integer and
bitwise aliases, signed 24-bit multiply/MAD operations, and `V_LERP_U8`, with
Vulkan/Metal translation tests and updated golden fixtures.

Validation completed on 2026-08-02:

- `git diff --check`
- `dotnet test SharpEmu.slnx --no-restore -v:minimal`
  - ShaderCompiler: 251 passed
  - ShaderCompiler.Metal: 202 passed
  - Libs: 849 passed, 1 transient AMPR temp-directory failure
- Isolated rerun of `AmprWriteAddressTests.CommandBufferWriteAddress0400_WritesValueOnCompletion`: passed
- Release CLI publish: completed successfully for `win-x64` and self-contained output
- NU1900 vulnerability-feed warnings are environmental (NuGet service unavailable)

Runnable Windows checkpoint:

`C:\Users\W10\Desktop\sharpemu-win64-29021b5\sharpemu-v72-isa-checkpoint-cli\SharpEmu.exe`

SHA-256:

`84BE82069D77401B4850D3113EF64D90CE9A98BB7F31C79C3DC56E18514643E8`

The repository's `.git` directory is read-only in this workspace, so the
checkpoint is recorded as this source marker and runnable publish folder
rather than a local Git commit.
