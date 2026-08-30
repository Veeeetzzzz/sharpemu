// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5DataShareSwizzleTests
{
    [Theory]
    [InlineData(0x041Fu, SpirvOp.BitwiseXor)] // bitmask SWAPX1
    [InlineData(0x80E4u, SpirvOp.ShiftRightLogical)] // quad identity
    [InlineData(0xC020u, SpirvOp.IAdd)] // rotate left by one
    [InlineData(0xE01Fu, SpirvOp.BitReverse)] // FFT, no-swizzle mask
    public void EveryDsSwizzleModeUsesSubgroupShuffleWithoutLds(
        uint pattern,
        SpirvOp modeOperation)
    {
        var swizzle = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Ds,
            "DsSwizzleB32",
            [],
            [Gen5Operand.Vector(1)],
            [Gen5Operand.Vector(2)],
            new Gen5DataShareControl(
                Offset0: pattern & 0xFF,
                Offset1: pattern >> 8,
                Gds: false));
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_E000, [swizzle, end]),
            [],
            null);
        var registers = new uint[256];
        var evaluation = new Gen5ShaderEvaluation(registers, registers, [], []);

        Assert.True(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state,
                evaluation,
                32,
                1,
                1,
                out var shader,
                out var error),
            error);

        var instructions = ReadInstructions(shader.Spirv);
        Assert.Equal(
            2,
            instructions.Count(item => item.Opcode == SpirvOp.GroupNonUniformShuffle));
        Assert.Contains(instructions, item => item.Opcode == modeOperation);
        Assert.DoesNotContain(
            instructions,
            item =>
                item.Opcode == SpirvOp.Variable &&
                item.Operands.Length >= 3 &&
                item.Operands[2] == (uint)SpirvStorageClass.Workgroup);
    }

    [Theory]
    [InlineData("DsBpermuteB32", 2)]
    [InlineData("DsPermuteB32", 96)]
    public void DsPermutesUseHalfWaveShufflesWithoutLds(
        string opcode,
        int expectedShuffleCount)
    {
        var permute = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Ds,
            opcode,
            [],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
            [Gen5Operand.Vector(2)],
            new Gen5DataShareControl(Offset0: 4, Offset1: 0, Gds: false));
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_E000, [permute, end]),
            [],
            null);
        var registers = new uint[256];
        var evaluation = new Gen5ShaderEvaluation(registers, registers, [], []);

        Assert.True(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state,
                evaluation,
                32,
                1,
                1,
                out var shader,
                out var error),
            error);

        var instructions = ReadInstructions(shader.Spirv);
        Assert.Equal(
            expectedShuffleCount,
            instructions.Count(item => item.Opcode == SpirvOp.GroupNonUniformShuffle));
        Assert.DoesNotContain(
            instructions,
            item =>
                item.Opcode == SpirvOp.Variable &&
                item.Operands.Length >= 3 &&
                item.Operands[2] == (uint)SpirvStorageClass.Workgroup);
    }

    [Theory]
    [InlineData("DsWriteB8", 2, 0, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsWriteB16", 2, 0, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsReadI8", 1, 1, SpirvOp.ShiftRightArithmetic)]
    [InlineData("DsReadU8", 1, 1, SpirvOp.ShiftRightLogical)]
    [InlineData("DsReadI16", 1, 1, SpirvOp.ShiftRightArithmetic)]
    [InlineData("DsReadU16", 1, 1, SpirvOp.ShiftLeftLogical)]
    [InlineData("DsWriteB64", 3, 0, SpirvOp.Store)]
    [InlineData("DsWrite2B64", 5, 0, SpirvOp.Store)]
    [InlineData("DsWrite2St64B64", 5, 0, SpirvOp.Store)]
    [InlineData("DsReadB64", 1, 2, SpirvOp.Load)]
    [InlineData("DsRead2B64", 1, 4, SpirvOp.Load)]
    [InlineData("DsRead2St64B64", 1, 4, SpirvOp.Load)]
    [InlineData("DsWriteB8D16Hi", 2, 0, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsWriteB16D16Hi", 2, 0, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsReadU8D16", 1, 1, SpirvOp.BitwiseAnd)]
    [InlineData("DsReadU8D16Hi", 1, 1, SpirvOp.ShiftLeftLogical)]
    [InlineData("DsReadI8D16", 1, 1, SpirvOp.ShiftRightArithmetic)]
    [InlineData("DsReadI8D16Hi", 1, 1, SpirvOp.ShiftRightArithmetic)]
    [InlineData("DsReadU16D16", 1, 1, SpirvOp.BitwiseOr)]
    [InlineData("DsReadU16D16Hi", 1, 1, SpirvOp.ShiftLeftLogical)]
    [InlineData("DsRsubU32", 2, 0, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsMskorB32", 3, 0, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsMskorRtnB32", 3, 1, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsIncU32", 2, 0, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsDecRtnU32", 2, 1, SpirvOp.AtomicCompareExchange)]
    [InlineData("DsWriteAddtidB32", 1, 0, SpirvOp.Store)]
    [InlineData("DsReadAddtidB32", 0, 1, SpirvOp.Load)]
    [InlineData("DsNop", 0, 0, SpirvOp.Variable)]
    public void CommonDsLoadsAndStoresCompileToSpirv(
        string opcode,
        int sourceCount,
        int destinationCount,
        SpirvOp expectedOperation)
    {
        var instruction = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Ds,
            opcode,
            [],
            Enumerable.Range(0, sourceCount)
                .Select(index => Gen5Operand.Vector((uint)index))
                .ToArray(),
            Enumerable.Range(0, destinationCount)
                .Select(index => Gen5Operand.Vector((uint)(16 + index)))
                .ToArray(),
            new Gen5DataShareControl(Offset0: 3, Offset1: 7, Gds: false));
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_E000, [instruction, end]),
            [],
            null);
        var registers = new uint[256];
        var evaluation = new Gen5ShaderEvaluation(registers, registers, [], []);

        Assert.True(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state,
                evaluation,
                32,
                1,
                1,
                out var shader,
                out var error),
            error);

        Assert.Contains(ReadInstructions(shader.Spirv), item => item.Opcode == expectedOperation);
    }

    private static IReadOnlyList<ParsedInstruction> ReadInstructions(byte[] spirv)
    {
        var instructions = new List<ParsedInstruction>();
        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var wordCount = checked((int)(header >> 16));
            Assert.InRange(wordCount, 1, (spirv.Length - offset) / sizeof(uint));
            var operands = new uint[wordCount - 1];
            for (var operand = 0; operand < operands.Length; operand++)
            {
                operands[operand] = BinaryPrimitives.ReadUInt32LittleEndian(
                    spirv.AsSpan(offset + ((operand + 1) * sizeof(uint))));
            }

            instructions.Add(new ParsedInstruction((SpirvOp)(ushort)header, operands));
            offset += wordCount * sizeof(uint);
        }

        return instructions;
    }

    private readonly record struct ParsedInstruction(SpirvOp Opcode, uint[] Operands);
}
