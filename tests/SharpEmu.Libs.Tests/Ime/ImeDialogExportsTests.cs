// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.HLE;
using SharpEmu.Libs.Ime;
using Xunit;

namespace SharpEmu.Libs.Tests.Ime;

public sealed class ImeDialogExportsTests
{
    private const ulong MemoryBase = 0x1_0000_0000;
    private const ulong ParamAddress = MemoryBase;
    private const ulong TextBufferAddress = MemoryBase + 0x100;
    private const ulong ResultAddress = MemoryBase + 0x300;

    private const int ParamSize = 0x60;
    private const int ParamMaxTextLengthOffset = 0x24;
    private const int ParamInputTextBufferOffset = 0x28;

    private const int StatusFinished = 2;

    public ImeDialogExportsTests()
    {
        ImeDialogExports.ResetForTests();
        Environment.SetEnvironmentVariable("SHARPEMU_IME_TEXT", null);
    }

    private static (FakeCpuMemory Memory, CpuContext Context) CreateDialog(uint maxTextLength = 16)
    {
        var memory = new FakeCpuMemory(MemoryBase, 0x400);
        var context = new CpuContext(memory, Generation.Gen5);

        Span<byte> param = stackalloc byte[ParamSize];
        param.Clear();
        BinaryPrimitives.WriteUInt32LittleEndian(param[ParamMaxTextLengthOffset..], maxTextLength);
        BinaryPrimitives.WriteUInt64LittleEndian(param[ParamInputTextBufferOffset..], TextBufferAddress);
        Assert.True(memory.TryWrite(ParamAddress, param));

        context[CpuRegister.Rdi] = ParamAddress;
        return (memory, context);
    }

    private static string ReadGuestText(FakeCpuMemory memory, int maxCharacters)
    {
        var bytes = new byte[(maxCharacters + 1) * sizeof(char)];
        Assert.True(memory.TryRead(TextBufferAddress, bytes));
        var text = Encoding.Unicode.GetString(bytes);
        var terminator = text.IndexOf('\0');
        return terminator < 0 ? text : text[..terminator];
    }

    [Fact]
    public void InitWritesConfiguredTextIntoTheGuestBuffer()
    {
        Environment.SetEnvironmentVariable("SHARPEMU_IME_TEXT", "Boletaria");
        var (memory, context) = CreateDialog();

        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));
        Assert.Equal("Boletaria", ReadGuestText(memory, 16));
    }

    [Fact]
    public void TextLongerThanMaxTextLengthIsTruncatedWithRoomForTerminator()
    {
        Environment.SetEnvironmentVariable("SHARPEMU_IME_TEXT", "AstraeaMaidenInBlack");
        var (memory, context) = CreateDialog(maxTextLength: 4);

        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));
        Assert.Equal("Ast", ReadGuestText(memory, 4));
    }

    [Theory]
    [InlineData(1u)]
    [InlineData(2u)]
    [InlineData(8u)]
    [InlineData(16u)]
    public void InitNeverWritesPastMaxTextLengthWideCharacters(uint maxTextLength)
    {
        Environment.SetEnvironmentVariable("SHARPEMU_IME_TEXT", "AstraeaMaidenInBlack");
        var memory = new FakeCpuMemory(MemoryBase, 0x400);
        var context = new CpuContext(memory, Generation.Gen5);

        var guard = new byte[(maxTextLength + 1) * sizeof(char)];
        Array.Fill(guard, (byte)0xEE);
        Assert.True(memory.TryWrite(TextBufferAddress, guard));

        Span<byte> param = stackalloc byte[ParamSize];
        param.Clear();
        BinaryPrimitives.WriteUInt32LittleEndian(param[ParamMaxTextLengthOffset..], maxTextLength);
        BinaryPrimitives.WriteUInt64LittleEndian(param[ParamInputTextBufferOffset..], TextBufferAddress);
        Assert.True(memory.TryWrite(ParamAddress, param));

        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));

        Span<byte> pastEnd = stackalloc byte[sizeof(char)];
        Assert.True(memory.TryRead(TextBufferAddress + (maxTextLength * sizeof(char)), pastEnd));
        Assert.Equal(0xEE, pastEnd[0]);
        Assert.Equal(0xEE, pastEnd[1]);
    }

    [Fact]
    public void StatusAdvancesFromRunningToFinishedOnTheFirstPoll()
    {
        var (_, context) = CreateDialog();
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));

        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(context));
        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(context));
    }

    [Fact]
    public void GetResultReportsSuccessOnlyAfterTheDialogFinishes()
    {
        var (memory, context) = CreateDialog();
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));

        context[CpuRegister.Rdi] = ResultAddress;
        Assert.NotEqual(0, ImeDialogExports.ImeDialogGetResult(context));

        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(context));
        Assert.Equal(0, ImeDialogExports.ImeDialogGetResult(context));

        Span<byte> endStatus = stackalloc byte[sizeof(int)];
        Assert.True(memory.TryRead(ResultAddress, endStatus));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(endStatus));
    }

    [Fact]
    public void GetResultWritesEndStatusOnly()
    {
        var (memory, context) = CreateDialog();
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));
        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(context));

        var poison = new byte[0x40];
        Array.Fill(poison, (byte)0xEE);
        Assert.True(memory.TryWrite(ResultAddress, poison));

        context[CpuRegister.Rdi] = ResultAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogGetResult(context));

        var readBack = new byte[0x40];
        Assert.True(memory.TryRead(ResultAddress, readBack));
        Assert.Equal(0, BinaryPrimitives.ReadInt32LittleEndian(readBack));
        Assert.All(readBack[sizeof(int)..], b => Assert.Equal(0xEE, b));
    }

    [Fact]
    public void InitRejectsASecondDialogWhileOneIsRunning()
    {
        var (_, context) = CreateDialog();
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));
        Assert.NotEqual(0, ImeDialogExports.ImeDialogInit(context));
    }

    [Fact]
    public void InitRejectsNullParametersAndBuffers()
    {
        var memory = new FakeCpuMemory(MemoryBase, 0x400);
        var context = new CpuContext(memory, Generation.Gen5);
        context[CpuRegister.Rdi] = 0;
        Assert.NotEqual(0, ImeDialogExports.ImeDialogInit(context));

        Span<byte> param = stackalloc byte[ParamSize];
        param.Clear();
        BinaryPrimitives.WriteUInt32LittleEndian(param[ParamMaxTextLengthOffset..], 16);
        Assert.True(memory.TryWrite(ParamAddress, param));
        context[CpuRegister.Rdi] = ParamAddress;
        Assert.NotEqual(0, ImeDialogExports.ImeDialogInit(context));
    }

    [Fact]
    public void TermAllowsAFreshDialogAndSecondTermIsRejected()
    {
        var (_, context) = CreateDialog();
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));
        Assert.Equal(StatusFinished, ImeDialogExports.ImeDialogGetStatus(context));
        Assert.Equal(0, ImeDialogExports.ImeDialogTerm(context));
        Assert.NotEqual(0, ImeDialogExports.ImeDialogTerm(context));

        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));
        Assert.NotEqual(0, ImeDialogExports.ImeDialogInit(context));
    }

    [Fact]
    public void AbortReportsAbortedEndStatus()
    {
        var (memory, context) = CreateDialog();
        Assert.Equal(0, ImeDialogExports.ImeDialogInit(context));
        Assert.Equal(0, ImeDialogExports.ImeDialogAbort(context));

        context[CpuRegister.Rdi] = ResultAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogGetResult(context));

        Span<byte> endStatus = stackalloc byte[sizeof(int)];
        Assert.True(memory.TryRead(ResultAddress, endStatus));
        Assert.Equal(2, BinaryPrimitives.ReadInt32LittleEndian(endStatus));
    }

    [Fact]
    public void ParamInitZeroesTheParameterBlock()
    {
        var memory = new FakeCpuMemory(MemoryBase, 0x400);
        var context = new CpuContext(memory, Generation.Gen5);
        var dirty = new byte[ParamSize];
        Array.Fill(dirty, (byte)0xCD);
        Assert.True(memory.TryWrite(ParamAddress, dirty));

        context[CpuRegister.Rdi] = ParamAddress;
        Assert.Equal(0, ImeDialogExports.ImeDialogParamInit(context));

        var readBack = new byte[ParamSize];
        Assert.True(memory.TryRead(ParamAddress, readBack));
        Assert.All(readBack, b => Assert.Equal(0, b));
    }
}
