// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5VectorRelativeMoveTests
{
    [Theory]
    [InlineData("VMovreldB32")]
    [InlineData("VMovrelsB32")]
    [InlineData("VMovrelsdB32")]
    [InlineData("VMovrelsd2B32")]
    public void RelativeMoveUsesM0ForBoundedDynamicVgprAccess(string opcode)
    {
        var move = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop1,
            opcode,
            [],
            [Gen5Operand.Vector(7)],
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            4,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [move, end]),
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
        Assert.Contains(instructions, item => item.Opcode == SpirvOp.IAdd);
        Assert.Contains(instructions, item => item.Opcode == SpirvOp.ULessThan);
        Assert.Contains(instructions, item => item.Opcode == SpirvOp.AccessChain);
    }

    [Theory]
    [InlineData("VMovrelsB32")]
    [InlineData("VMovrelsdB32")]
    [InlineData("VMovrelsd2B32")]
    public void RelativeSourceRejectsNonVgprBase(string opcode)
    {
        var move = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop1,
            opcode,
            [],
            [Gen5Operand.Scalar(7)],
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            4,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [move, end]),
            [],
            null);
        var registers = new uint[256];
        var evaluation = new Gen5ShaderEvaluation(registers, registers, [], []);

        Assert.False(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state,
                evaluation,
                32,
                1,
                1,
                out _,
                out var error));
        Assert.Contains("expects a VGPR source base", error, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("VSwapB32")]
    [InlineData("VSwaprelB32")]
    public void SwapReadsBothValuesBeforeWritingEither(string opcode)
    {
        var swap = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop1,
            opcode,
            [],
            [Gen5Operand.Vector(7), Gen5Operand.Vector(5)],
            [Gen5Operand.Vector(5), Gen5Operand.Vector(7)],
            null);
        var end = new Gen5ShaderInstruction(
            4,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F080, [swap, end]),
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
        Assert.True(instructions.Count(item => item.Opcode == SpirvOp.Load) >= 2);
        Assert.True(instructions.Count(item => item.Opcode == SpirvOp.Store) >= 2);
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
