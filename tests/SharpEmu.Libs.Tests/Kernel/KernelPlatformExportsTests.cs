// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Kernel;

public sealed class KernelPlatformExportsTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const ulong OpenPsIdAddress = MemoryBase + 0x40;

    [Fact]
    public void TrinityModeReportsBaseModel()
    {
        var context = new CpuContext(new FakeCpuMemory(MemoryBase, 0x100), Generation.Gen5);

        Assert.Equal(0, KernelExports.KernelIsTrinityMode(context));
        Assert.Equal(0UL, context[CpuRegister.Rax]);
    }

    [Fact]
    public void OpenPsIdWritesAStableZeroedSixteenByteValue()
    {
        var memory = new FakeCpuMemory(MemoryBase, 0x100);
        var context = new CpuContext(memory, Generation.Gen5);
        context[CpuRegister.Rdi] = OpenPsIdAddress;

        Assert.Equal(0, KernelExports.KernelGetOpenPsId(context));
        Assert.Equal(0UL, context[CpuRegister.Rax]);

        Span<byte> actual = stackalloc byte[16];
        Assert.True(memory.TryRead(OpenPsIdAddress, actual));
        Assert.All(actual.ToArray(), value => Assert.Equal((byte)0, value));
    }

    [Fact]
    public void OpenPsIdRejectsNullAndUnreadableBuffers()
    {
        var memory = new FakeCpuMemory(MemoryBase, 0x100);
        var context = new CpuContext(memory, Generation.Gen5);

        context[CpuRegister.Rdi] = 0;
        Assert.Equal(
            (int)OrbisGen2Result.ORBIS_GEN2_ERROR_INVALID_ARGUMENT,
            KernelExports.KernelGetOpenPsId(context));

        context[CpuRegister.Rdi] = MemoryBase + 0xF8;
        Assert.Equal(
            (int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT,
            KernelExports.KernelGetOpenPsId(context));
    }
}
