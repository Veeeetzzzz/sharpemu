// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5ScalarSingleSourceOpcodeTests
{
    [Theory]
    [InlineData("SAbsI32", 0x8000_0000u, 0x8000_0000u)]
    [InlineData("SAbsI32", 0xFFFF_FFF0u, 0x0000_0010u)]
    [InlineData("SFlbitI32B32", 0x0000_0010u, 27u)]
    [InlineData("SFlbitI32B32", 0u, 0xFFFF_FFFFu)]
    public void SingleSource32OpcodesFoldWithoutASecondOperand(
        string opcode,
        uint source,
        uint expected)
    {
        var instruction = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Sop1,
            opcode,
            [],
            [Literal(source)],
            [Gen5Operand.Scalar(4)],
            null);

        var evaluation = Evaluate([instruction]);
        Assert.Equal(expected, evaluation.ScalarRegisters[4]);
    }

    [Theory]
    [InlineData("SFF1I32B64", 0x0001_0000u, 0x0000_0000u, 16u)]
    [InlineData("SBcnt1I32B64", 0x0000_F0F0u, 0x0000_0000u, 8u)]
    [InlineData("SFF1I32B64", 0u, 0u, 0xFFFF_FFFFu)]
    public void SingleSource64OpcodesReadARegisterPair(
        string opcode,
        uint low,
        uint high,
        uint expected)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            Move(0, 0, low),
            Move(4, 1, high),
            new(
                8,
                Gen5ShaderEncoding.Sop1,
                opcode,
                [],
                [Gen5Operand.Scalar(0)],
                [Gen5Operand.Scalar(4)],
                null),
        };

        var evaluation = Evaluate(instructions);
        Assert.Equal(expected, evaluation.ScalarRegisters[4]);
    }

    private static Gen5ShaderEvaluation Evaluate(IReadOnlyList<Gen5ShaderInstruction> instructions)
    {
        var program = instructions
            .Concat(
            [
                new Gen5ShaderInstruction(
                    (uint)(instructions.Count * sizeof(uint)),
                    Gen5ShaderEncoding.Sopp,
                    "SEndpgm",
                    [0xBF810000u],
                    [],
                    [],
                    null),
            ])
            .ToArray();
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0001_3000, program),
            [],
            null);
        var context = new CpuContext(new EmptyCpuMemory(), Generation.Gen5);

        Assert.True(
            Gen5ShaderScalarEvaluator.TryEvaluate(
                context,
                state,
                out var evaluation,
                out var error),
            error);
        return evaluation;
    }

    private static Gen5ShaderInstruction Move(uint pc, uint destination, uint literal) =>
        new(
            pc,
            Gen5ShaderEncoding.Sop1,
            "SMovB32",
            [],
            [Literal(literal)],
            [Gen5Operand.Scalar(destination)],
            null);

    private static Gen5Operand Literal(uint value) =>
        new(Gen5OperandKind.LiteralConstant, value);

    private sealed class EmptyCpuMemory : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => false;

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;
    }
}
