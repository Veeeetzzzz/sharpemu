// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5PackedIntegerTests
{
    [Theory]
    [InlineData("VPkMadI16")]
    [InlineData("VPkMulLoU16")]
    [InlineData("VPkAddI16")]
    [InlineData("VPkSubI16")]
    [InlineData("VPkLshlrevB16")]
    [InlineData("VPkLshrrevB16")]
    [InlineData("VPkAshrrevI16")]
    [InlineData("VPkMaxI16")]
    [InlineData("VPkMinI16")]
    [InlineData("VPkMadU16")]
    [InlineData("VPkAddU16")]
    [InlineData("VPkSubU16")]
    [InlineData("VPkMaxU16")]
    [InlineData("VPkMinU16")]
    public void PackedIntegerFamilyEmitsValidSpirv(string opcode)
    {
        var packed = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3p,
            opcode,
            [],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(5)],
            new Gen5Vop3pControl(
                OpSelMask: 5,
                OpSelHiMask: 2,
                NegLoMask: 1,
                NegHiMask: 4,
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
            new Gen5ShaderProgram(0x1_0000_F800, [packed, end]),
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
        Assert.NotEmpty(shader.Spirv);
    }

    [Theory]
    [InlineData("VDot2I32I16")]
    [InlineData("VDot2U32U16")]
    [InlineData("VDot4I32I8")]
    [InlineData("VDot4U32U8")]
    [InlineData("VDot8I32I4")]
    [InlineData("VDot8U32U4")]
    public void PackedIntegerDotFamilyEmitsValidSpirv(string opcode)
    {
        var dot = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3p,
            opcode,
            [],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(5)],
            new Gen5Vop3pControl(0, 0, 0, 0, Clamp: true));
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_FA00, [dot, end]),
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
        Assert.NotEmpty(shader.Spirv);
    }

    [Fact]
    public void PackedFloatDotEmitsValidSpirvWithDocumentedModifierDomains()
    {
        var dot = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop3p,
            "VDot2F32F16",
            [],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            [Gen5Operand.Vector(5)],
            new Gen5Vop3pControl(
                OpSelMask: 5,
                OpSelHiMask: 2,
                NegLoMask: 5,
                NegHiMask: 5,
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
            new Gen5ShaderProgram(0x1_0000_FB00, [dot, end]),
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
        Assert.NotEmpty(shader.Spirv);
    }
}
