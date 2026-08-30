// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5VectorCompareTests
{
    [Theory]
    [InlineData("VCmpLtF16", SpirvOp.FOrdLessThan)]
    [InlineData("VCmpLtI16", SpirvOp.SLessThan)]
    [InlineData("VCmpLtU16", SpirvOp.ULessThan)]
    [InlineData("VCmpxNgeF16", SpirvOp.FUnordLessThan)]
    [InlineData("VCmpLtF64", SpirvOp.ULessThan)]
    [InlineData("VCmpxGtF64", SpirvOp.UGreaterThan)]
    [InlineData("VCmpLtI64", SpirvOp.SLessThan)]
    [InlineData("VCmpEqU64", SpirvOp.IEqual)]
    [InlineData("VCmpxGeU64", SpirvOp.UGreaterThanEqual)]
    public void SixteenBitCompareFamiliesEmitTypedPredicates(
        string opcode,
        SpirvOp expectedOperation)
    {
        var compare = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vopc,
            opcode,
            [],
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
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
            new Gen5ShaderProgram(0x1_0000_F000, [compare, end]),
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
        Assert.Contains(expectedOperation, ReadOpcodes(shader.Spirv));
    }

    [Theory]
    [InlineData("VLshlrevB64", SpirvOp.ShiftLeftLogical)]
    [InlineData("VLshrrevB64", SpirvOp.ShiftRightLogical)]
    [InlineData("VAshrrevI64", SpirvOp.ShiftRightArithmetic)]
    public void Gfx10Vector64ShiftsEmitTypedPairOperations(
        string opcode,
        SpirvOp expectedOperation)
    {
        var shift = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            opcode,
            [],
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            12,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [shift, end]),
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
        Assert.Contains(expectedOperation, ReadOpcodes(shader.Spirv));
    }

    [Theory]
    [InlineData("VMulI32I24", SpirvOp.IMul)]
    [InlineData("VMulHiI32I24", SpirvOp.ShiftRightArithmetic)]
    [InlineData("VMadI32I24", SpirvOp.IAdd)]
    public void Gfx10Signed24FamiliesEmitWideArithmetic(
        string opcode,
        SpirvOp expectedOperation)
    {
        var encoding = opcode == "VMadI32I24"
            ? Gen5ShaderEncoding.Vop3
            : Gen5ShaderEncoding.Vop2;
        var sources = opcode == "VMadI32I24"
            ? new[] { Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3) }
            : new[] { Gen5Operand.Vector(1), Gen5Operand.Vector(2) };
        var operation = new Gen5ShaderInstruction(
            0,
            encoding,
            opcode,
            [],
            sources,
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            12,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [operation, end]),
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
        Assert.Contains(expectedOperation, ReadOpcodes(shader.Spirv));
    }

    [Fact]
    public void Gfx10LerpU8EmitsPackedByteAverages()
    {
        var operation = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            "VLerpU8",
            [],
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            12,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [operation, end]),
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
        Assert.Contains(SpirvOp.ShiftRightLogical, ReadOpcodes(shader.Spirv));
        Assert.Contains(SpirvOp.BitwiseOr, ReadOpcodes(shader.Spirv));
    }

    [Fact]
    public void Gfx10MsadU8EmitsMaskedByteAbsDifferences()
    {
        var operation = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            "VMsadU8",
            [],
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            12,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [operation, end]),
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
        var opcodes = ReadOpcodes(shader.Spirv);
        Assert.Contains(SpirvOp.INotEqual, opcodes);
        Assert.Contains(SpirvOp.Select, opcodes);
        Assert.Contains(SpirvOp.UGreaterThan, opcodes);
    }

    [Fact]
    public void Gfx10DivFmasF32SelectsVccScaledFusedResult()
    {
        var operation = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            "VDivFmasF32",
            [],
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            12,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [operation, end]),
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
        var opcodes = ReadOpcodes(shader.Spirv);
        Assert.Contains(SpirvOp.ExtInst, opcodes);
        Assert.Contains(SpirvOp.FMul, opcodes);
        Assert.Contains(SpirvOp.Select, opcodes);
    }

    [Theory]
    [InlineData("VMulLegacyF32", SpirvOp.Select)]
    [InlineData("VMacLegacyF32", SpirvOp.FAdd)]
    [InlineData("VMadLegacyF32", SpirvOp.FAdd)]
    [InlineData("VMullitF32", SpirvOp.Select)]
    public void Gfx10LegacyFloatFamiliesApplyDx9ZeroProductRules(
        string opcode,
        SpirvOp expectedOperation)
    {
        var encoding = opcode == "VMulLegacyF32"
            ? Gen5ShaderEncoding.Vop2
            : Gen5ShaderEncoding.Vop3;
        var sources = opcode == "VMadLegacyF32"
            ? new[] { Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3) }
            : new[] { Gen5Operand.Vector(1), Gen5Operand.Vector(2) };
        var operation = new Gen5ShaderInstruction(
            0,
            encoding,
            opcode,
            [],
            sources,
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            12,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [operation, end]),
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
        var opcodes = ReadOpcodes(shader.Spirv);
        Assert.Contains(SpirvOp.FMul, opcodes);
        Assert.Contains(SpirvOp.Select, opcodes);
        Assert.Contains(expectedOperation, opcodes);
    }

    [Theory]
    [InlineData("VQsadPkU16U8")]
    [InlineData("VMqsadPkU16U8")]
    [InlineData("VMqsadU32U8")]
    public void Gfx10PackedSadFamiliesEmitPairOrQuadStores(string opcode)
    {
        var operation = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3,
            opcode,
            [],
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            [Gen5Operand.Vector(5)],
            null);
        var end = new Gen5ShaderInstruction(
            12,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_F000, [operation, end]),
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
        var opcodes = ReadOpcodes(shader.Spirv);
        Assert.Contains(SpirvOp.IAdd, opcodes);
        Assert.Contains(SpirvOp.ShiftRightLogical, opcodes);
        Assert.Contains(SpirvOp.Store, opcodes);
    }

    private static IReadOnlyList<SpirvOp> ReadOpcodes(byte[] spirv)
    {
        var opcodes = new List<SpirvOp>();
        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var wordCount = checked((int)(header >> 16));
            Assert.InRange(wordCount, 1, (spirv.Length - offset) / sizeof(uint));
            opcodes.Add((SpirvOp)(ushort)header);
            offset += wordCount * sizeof(uint);
        }

        return opcodes;
    }
}
