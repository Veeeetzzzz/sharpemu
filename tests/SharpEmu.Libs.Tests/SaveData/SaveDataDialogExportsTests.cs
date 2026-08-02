// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.SaveData;
using Xunit;

namespace SharpEmu.Libs.Tests.SaveData;

public sealed class SaveDataDialogExportsTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const ulong ParamAddress = MemoryBase;
    private const ulong ResultAddress = MemoryBase + 0x200;

    private const int StatusRunning = 2;
    private const int StatusFinished = 3;

    public SaveDataDialogExportsTests() => SaveDataDialogExports.ResetForTests();

    private static CpuContext CreateContext(out FakeCpuMemory memory)
    {
        memory = new FakeCpuMemory(MemoryBase, 0x400);
        return new CpuContext(memory, Generation.Gen5);
    }

    [Fact]
    public void RepeatedInitializeAlwaysSucceeds()
    {
        var context = CreateContext(out _);

        for (var attempt = 0; attempt < 5; attempt++)
        {
            Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));
            Assert.Equal(0UL, context[CpuRegister.Rax]);
        }
    }

    [Fact]
    public void FirstPollReportsRunningBeforeTheDialogFinishes()
    {
        var context = CreateContext(out _);
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));

        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogOpen(context));

        Assert.Equal(StatusRunning, SaveDataDialogExports.SaveDataDialogGetStatus(context));
        Assert.Equal(StatusFinished, SaveDataDialogExports.SaveDataDialogGetStatus(context));
        Assert.Equal(StatusFinished, SaveDataDialogExports.SaveDataDialogGetStatus(context));
    }

    [Fact]
    public void ReinitializingWhileRunningDoesNotResetTheDialog()
    {
        var context = CreateContext(out _);
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));

        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogOpen(context));
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));

        Assert.Equal(StatusRunning, SaveDataDialogExports.SaveDataDialogGetStatus(context));
        Assert.Equal(StatusFinished, SaveDataDialogExports.SaveDataDialogGetStatus(context));
    }

    [Fact]
    public void ReopeningReArmsTheRunningPoll()
    {
        var context = CreateContext(out _);
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));

        for (var cycle = 0; cycle < 3; cycle++)
        {
            context[CpuRegister.Rdi] = ParamAddress;
            Assert.Equal(0, SaveDataDialogExports.SaveDataDialogOpen(context));
            Assert.Equal(StatusRunning, SaveDataDialogExports.SaveDataDialogGetStatus(context));
            Assert.Equal(StatusFinished, SaveDataDialogExports.SaveDataDialogGetStatus(context));
        }
    }

    [Fact]
    public void FullOpenPollResultCycleReportsTheAffirmativeButton()
    {
        var context = CreateContext(out var memory);
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));

        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogOpen(context));
        Assert.Equal(StatusRunning, SaveDataDialogExports.SaveDataDialogGetStatus(context));
        Assert.Equal(StatusFinished, SaveDataDialogExports.SaveDataDialogGetStatus(context));

        context[CpuRegister.Rdi] = ResultAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogGetResult(context));

        Span<byte> buttonId = stackalloc byte[sizeof(int)];
        Assert.True(memory.TryRead(ResultAddress + 0x08, buttonId));
        Assert.Equal(1, BinaryPrimitives.ReadInt32LittleEndian(buttonId));
    }

    // A captured OrbisSaveDataDialogParam: +0x00 is the common-header size
    // (0x30), while the requested mode is at +0x34.
    private static readonly byte[] CapturedParam = Convert.FromHexString(
        "30000000000000000000000000000000" +
        "00000000000000000000000000000000" +
        "00000000000000000000000069932BC9" +
        "98000000030000000100000000000000" +
        "60FE590806000000C0FD590806000000" +
        "000000000000000080FD590806000000");

    [Fact]
    public void ModeIsReadFromTheDialogFieldNotTheBaseParamSize()
    {
        var context = CreateContext(out var memory);
        Assert.True(memory.TryWrite(ParamAddress, CapturedParam));
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));

        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogOpen(context));
        Assert.Equal(StatusRunning, SaveDataDialogExports.SaveDataDialogGetStatus(context));
        Assert.Equal(StatusFinished, SaveDataDialogExports.SaveDataDialogGetStatus(context));

        context[CpuRegister.Rdi] = ResultAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogGetResult(context));

        Span<byte> mode = stackalloc byte[sizeof(int)];
        Assert.True(memory.TryRead(ResultAddress, mode));
        Assert.Equal(3, BinaryPrimitives.ReadInt32LittleEndian(mode));
        Assert.NotEqual(48, BinaryPrimitives.ReadInt32LittleEndian(mode));
    }

    [Fact]
    public void UserDataOutsideDeclaredStructSizeIsReportedAsNull()
    {
        var context = CreateContext(out var memory);
        Assert.True(memory.TryWrite(ParamAddress, CapturedParam));

        var poison = new byte[0x40];
        Array.Fill(poison, (byte)0xEE);
        Assert.True(memory.TryWrite(ParamAddress + 0xC8, poison));

        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));
        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogOpen(context));
        Assert.Equal(StatusRunning, SaveDataDialogExports.SaveDataDialogGetStatus(context));
        Assert.Equal(StatusFinished, SaveDataDialogExports.SaveDataDialogGetStatus(context));

        context[CpuRegister.Rdi] = ResultAddress;
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogGetResult(context));

        Span<byte> userData = stackalloc byte[sizeof(ulong)];
        Assert.True(memory.TryRead(ResultAddress + 0x20, userData));
        Assert.Equal(0UL, BinaryPrimitives.ReadUInt64LittleEndian(userData));
    }

    [Fact]
    public void GetResultBeforeTheDialogFinishesIsRejected()
    {
        var context = CreateContext(out _);
        Assert.Equal(0, SaveDataDialogExports.SaveDataDialogInitialize(context));

        context[CpuRegister.Rdi] = ResultAddress;
        Assert.NotEqual(0, SaveDataDialogExports.SaveDataDialogGetResult(context));
    }
}
