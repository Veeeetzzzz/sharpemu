# SharpEmu v71 source checkpoint

This checkpoint contains the Gen5/GFX10 shader coverage pass through the 64-bit
vector shift and integer/bitwise alias families, plus the earlier Vulkan/Metal
video lifetime and detile fixes.

Validation completed on 2026-08-01:

- `dotnet test SharpEmu.slnx --no-restore -v:minimal`
- 243 ShaderCompiler tests passed
- 198 ShaderCompiler.Metal tests passed
- 850 Libs tests passed
- NU1900 package-vulnerability lookup warning remains environmental

Runnable Windows checkpoint:

`C:\Users\W10\Desktop\sharpemu-win64-29021b5\sharpemu-v71-isa-checkpoint-cli\SharpEmu.exe`

SHA-256:

`2E653FEB595549FD5EF0FDFE2C2A896C6118C6239DADE9E0CAEE709B12F6CCDE`

The repository's `.git` directory is read-only in this workspace, so the
checkpoint is recorded as this source marker and runnable publish folder rather
than a local Git commit.
