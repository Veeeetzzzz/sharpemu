// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5ScalarRelativeMoveTests
{
    private const uint LowValue = 0x11223344;
    private const uint HighValue = 0xAABBCCDD;

    [Theory]
    [InlineData("SMovrelsB32")]
    [InlineData("SMovrelsB64")]
    [InlineData("SMovreldB32")]
    [InlineData("SMovreldB64")]
    public void ScalarRelativeMoveUsesBoundedM0Indexing(string opcode)
    {
        var move = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Sop1,
            opcode,
            [],
            [Gen5Operand.Scalar(7)],
            [Gen5Operand.Scalar(5)],
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
            new Gen5ShaderProgram(0x1_0001_0000, [move, end]),
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
    [InlineData("SMovrelsB32", false, true)]
    [InlineData("SMovrelsB64", true, true)]
    [InlineData("SMovreldB32", false, false)]
    [InlineData("SMovreldB64", true, false)]
    public void ScalarEvaluatorExecutesRelativeMove(
        string opcode,
        bool is64,
        bool relativeSource)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            MoveScalar(0, 124, Gen5Operand.Source(129)), // m0 = 1
        };
        var sourceBase = 7u;
        var sourceIndex = relativeSource ? sourceBase + 1 : sourceBase;
        instructions.Add(MoveScalar(4, sourceIndex, Literal(LowValue)));
        if (is64)
        {
            instructions.Add(MoveScalar(8, sourceIndex + 1, Literal(HighValue)));
        }

        var movePc = (uint)(instructions.Count * sizeof(uint));
        instructions.Add(new Gen5ShaderInstruction(
            movePc,
            Gen5ShaderEncoding.Sop1,
            opcode,
            [],
            [Gen5Operand.Scalar(sourceBase)],
            [Gen5Operand.Scalar(5)],
            null));
        instructions.Add(new Gen5ShaderInstruction(
            movePc + sizeof(uint),
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null));

        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0001_1000, instructions),
            [],
            null);
        var ctx = new CpuContext(new EmptyCpuMemory(), Generation.Gen5);
        Assert.True(
            Gen5ShaderScalarEvaluator.TryEvaluate(
                ctx,
                state,
                out var evaluation,
                out var error),
            error);

        var destination = relativeSource ? 5u : 6u;
        Assert.Equal(LowValue, evaluation.ScalarRegisters[(int)destination]);
        if (is64)
        {
            Assert.Equal(HighValue, evaluation.ScalarRegisters[(int)destination + 1]);
        }
    }

    [Theory]
    [InlineData("SClause")]
    [InlineData("SWaitcntDepctr")]
    public void SchedulingOnlySoppInstructionCompilesAsNoOp(string opcode)
    {
        var scheduling = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Sopp,
            opcode,
            [],
            [],
            [],
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
            new Gen5ShaderProgram(0x1_0001_2000, [scheduling, end]),
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
                out _,
                out var error),
            error);
    }

    private static Gen5ShaderInstruction MoveScalar(
        uint pc,
        uint destination,
        Gen5Operand source) =>
        new(
            pc,
            Gen5ShaderEncoding.Sop1,
            "SMovB32",
            [],
            [source],
            [Gen5Operand.Scalar(destination)],
            null);

    private static Gen5Operand Literal(uint value) =>
        new(Gen5OperandKind.LiteralConstant, value);

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

    private sealed class EmptyCpuMemory : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => false;

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;
    }
}
