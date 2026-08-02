# SharpEmu v73 source checkpoint

This checkpoint packages the current source worktree after the RDNA2/GFX10
ISA coverage pass. In addition to the earlier vector compare/shift, integer,
signed-24-bit, and `V_LERP_U8` work, this pass adds `V_MSAD_U8`,
`V_QSAD_PK_U16_U8`, `V_MQSAD_PK_U16_U8`, and `V_MQSAD_U32_U8` decoding and
Vulkan/Metal translation with focused tests.

Validation completed on 2026-08-02:

- `git diff --check`
- `dotnet test SharpEmu.slnx --no-restore -v:minimal`
  - ShaderCompiler: 259 passed
  - ShaderCompiler.Metal: 206 passed
  - Libs: 850 passed
- Release CLI publish completed successfully for `win-x64` and self-contained output
- NU1900 vulnerability-feed warning is environmental (NuGet service unavailable)

Runnable Windows checkpoint:

`C:\Users\W10\Desktop\sharpemu-win64-29021b5\sharpemu-v73-sad-checkpoint-cli\SharpEmu.exe`

SHA-256:

`7D3462E1C828F2C93A126CBADA946CFD2503136415ADC6851AACF052ED4669DC`

The repository's `.git` directory is read-only in this workspace, so the
checkpoint is recorded as this source marker and runnable publish folder
rather than a local Git commit.
