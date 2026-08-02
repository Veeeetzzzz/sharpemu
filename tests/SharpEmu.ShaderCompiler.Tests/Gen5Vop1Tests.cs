// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5Vop1Tests
{
    [Theory]
    [InlineData("VFfbhU32")]
    [InlineData("VFfbhI32")]
    [InlineData("VCvtF16U16")]
    [InlineData("VCvtF16I16")]
    [InlineData("VCvtU16F16")]
    [InlineData("VCvtI16F16")]
    [InlineData("VRcpF16")]
    [InlineData("VSqrtF16")]
    [InlineData("VRsqF16")]
    [InlineData("VLogF16")]
    [InlineData("VExpF16")]
    [InlineData("VFrexpMantF16")]
    [InlineData("VFrexpExpI16F16")]
    [InlineData("VFloorF16")]
    [InlineData("VCeilF16")]
    [InlineData("VTruncF16")]
    [InlineData("VRndneF16")]
    [InlineData("VFractF16")]
    [InlineData("VSinF16")]
    [InlineData("VCosF16")]
    [InlineData("VSatPkU8I16")]
    [InlineData("VCvtNormI16F16")]
    [InlineData("VCvtNormU16F16")]
    [InlineData("VFrexpExpI32F32")]
    [InlineData("VFrexpMantF32")]
    [InlineData("VFrexpExpI32F64")]
    [InlineData("VFrexpMantF64")]
    [InlineData("VCvtI32F64")]
    [InlineData("VCvtF64I32")]
    [InlineData("VCvtF32F64")]
    [InlineData("VCvtF64F32")]
    [InlineData("VCvtU32F64")]
    [InlineData("VCvtF64U32")]
    [InlineData("VTruncF64")]
    [InlineData("VCeilF64")]
    [InlineData("VRndneF64")]
    [InlineData("VFloorF64")]
    [InlineData("VFractF64")]
    public void DocumentedGfx10Vop1FamilyEmitsSpirv(string opcode)
    {
        var floatingOrHalf = opcode.Contains("F16", StringComparison.Ordinal) &&
                             opcode is not "VSatPkU8I16";
        var instruction = new Gen5ShaderInstruction(
            0,
            floatingOrHalf ? Gen5ShaderEncoding.Vop3 : Gen5ShaderEncoding.Vop1,
            opcode,
            [],
            [Gen5Operand.Vector(1)],
            [Gen5Operand.Vector(5)],
            floatingOrHalf
                ? new Gen5Vop3Control(
                    AbsoluteMask: opcode is "VCvtF16U16" or "VCvtF16I16" ? 0u : 1u,
                    NegateMask: 0,
                    OutputModifier: 1,
                    Clamp: true,
                    OperandSelect: 9,
                    ScalarDestination: null)
                : null);
        var end = new Gen5ShaderInstruction(
            8,
            Gen5ShaderEncoding.Sopp,
            "SEndpgm",
            [0xBF810000],
            [],
            [],
            null);
        var state = new Gen5ShaderState(
            new Gen5ShaderProgram(0x1_0000_FC00, [instruction, end]),
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
    [InlineData("VPipeflush")]
    [InlineData("VClrexcp")]
    public void StateOnlyVop1InstructionsCompileAsNoOps(string opcode)
    {
        var instruction = new Gen5ShaderInstruction(
            0,
            Gen5ShaderEncoding.Vop1,
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
            new Gen5ShaderProgram(0x1_0000_FD00, [instruction, end]),
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
