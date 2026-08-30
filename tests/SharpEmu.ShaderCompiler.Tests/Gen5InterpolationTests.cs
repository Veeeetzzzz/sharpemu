// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5InterpolationTests
{
    [Fact]
    public void InterpMovLoadsTheDeclaredPixelInput()
    {
        var interpolation = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vintrp,
            "VInterpMovF32",
            [0xC8020002],
            [Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(0)],
            new Gen5InterpolationControl(Attribute: 0, Channel: 0));
        var end = new Gen5ShaderInstruction(
            4,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_D000, [interpolation, end]),
            [],
            null);
        var registers = new uint[256];
        var evaluation = new Gen5ShaderEvaluation(registers, registers, [], []);

        Assert.True(
            Gen5SpirvTranslator.TryCompilePixelShader(
                state,
                evaluation,
                Gen5PixelOutputKind.Float,
                out var shader,
                out var error),
            error);

        var instructions = ReadInstructions(shader.Spirv);
        var location = Assert.Single(
            instructions,
            item =>
                item.Opcode == SpirvOp.Decorate &&
                item.Operands.Length >= 3 &&
                item.Operands[1] == (uint)SpirvDecoration.Location &&
                item.Operands[2] == 0 &&
                instructions.Any(variable =>
                    variable.Opcode == SpirvOp.Variable &&
                    variable.Operands.Length >= 3 &&
                    variable.Operands[1] == item.Operands[0] &&
                    variable.Operands[2] == (uint)SpirvStorageClass.Input));
        var inputVariable = location.Operands[0];
        var inputLoad = Assert.Single(
            instructions,
            item =>
                item.Opcode == SpirvOp.Load &&
                item.Operands.Length >= 3 &&
                item.Operands[2] == inputVariable);
        var inputValue = inputLoad.Operands[1];
        Assert.Contains(
            instructions,
            item =>
                item.Opcode == SpirvOp.CompositeExtract &&
                item.Operands.Length >= 4 &&
                item.Operands[2] == inputValue &&
                item.Operands[3] == 0);
    }

    [Fact]
    public void ExtendedHalfInterpolationPacksSelectedDestinationHalf()
    {
        var interpolation = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            "VInterpP2F16",
            [0xD75AC005, 0x0C0E05C0],
            [Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            [Gen5Operand.Vector(5)],
            new Gen5InterpolationControl(
                Attribute: 0,
                Channel: 3,
                AttributeWordHigh: true,
                HalfPrecisionResult: true,
                DestinationHigh: true,
                OutputModifier: 1,
                Clamp: true));
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_D800, [interpolation, end]),
            [],
            null);
        var registers = new uint[256];
        var evaluation = new Gen5ShaderEvaluation(registers, registers, [], []);

        Assert.True(
            Gen5SpirvTranslator.TryCompilePixelShader(
                state,
                evaluation,
                Gen5PixelOutputKind.Float,
                out var shader,
                out var error),
            error);

        var opcodes = ReadInstructions(shader.Spirv).Select(item => item.Opcode).ToArray();
        Assert.Contains(SpirvOp.FMul, opcodes);
        Assert.Contains(SpirvOp.BitwiseAnd, opcodes);
        Assert.Contains(SpirvOp.ShiftLeftLogical, opcodes);
        Assert.Contains(SpirvOp.BitwiseOr, opcodes);
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
