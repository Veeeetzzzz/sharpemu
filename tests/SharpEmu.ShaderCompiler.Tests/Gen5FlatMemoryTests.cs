// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5FlatMemoryTests
{
    private const ulong ShaderAddress = 0x1_0000_0000;
    private const uint SEndpgm = 0xBF810000;

    [Theory]
    [InlineData(0x7D920501u, "VCmpLtF16")]
    [InlineData(0x7D120501u, "VCmpLtI16")]
    [InlineData(0x7D520501u, "VCmpLtU16")]
    [InlineData(0x7DB20501u, "VCmpxLtF16")]
    [InlineData(0x7D320501u, "VCmpxLtI16")]
    [InlineData(0x7D720501u, "VCmpxLtU16")]
    [InlineData(0x7D1E0501u, "VCmpClassF16")]
    [InlineData(0x7C420501u, "VCmpLtF64")]
    [InlineData(0x7C620501u, "VCmpxLtF64")]
    [InlineData(0x7D440501u, "VCmpEqI64")]
    [InlineData(0x7DC40501u, "VCmpEqU64")]
    [InlineData(0x7D6404C1u, "VCmpxEqI64")]
    [InlineData(0x7DE404C1u, "VCmpxEqU64")]
    public void Gfx10SixteenBitCompareEncodingsDecode(uint word, string expectedOpcode)
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x100);
        Span<byte> shader = stackalloc byte[2 * sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(shader, word);
        BinaryPrimitives.WriteUInt32LittleEndian(shader[sizeof(uint)..], SEndpgm);
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var error),
            error);
        Assert.Equal(expectedOpcode, program.Instructions[0].Opcode);
    }

    [Fact]
    public void Gfx10ExtendedCompareEncodingUsesScalarDestinationAndTwoSources()
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x100);
        uint[] words =
        [
            0xD4D9007Eu, // v_cmpx_lt_f16_e64 v1, v2
            0x02020501u,
            SEndpgm,
        ];
        Span<byte> shader = stackalloc byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                shader[(index * sizeof(uint))..],
                words[index]);
        }
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var error),
            error);

        var compare = program.Instructions[0];
        Assert.Equal("VCmpxLtF16", compare.Opcode);
        Assert.Equal([Gen5Operand.Vector(1), Gen5Operand.Vector(2)], compare.Sources);
        Assert.Equal([Gen5Operand.Scalar(126)], compare.Destinations);
        var control = Assert.IsType<Gen5Vop3Control>(compare.Control);
        Assert.Equal(126u, control.ScalarDestination);
    }

    [Fact]
    public void Gfx10ExtendedInteger64CompareUsesPairSources()
    {
        var instruction = DecodeFirst(
            0xD4B3007Eu, // v_cmpx_le_i64_e64 v[1:2], v[2:3]
            0x02020501u,
            SEndpgm);

        Assert.Equal("VCmpxLeI64", instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Scalar(126)], instruction.Destinations);
    }

    [Theory]
    [InlineData(0xD6FF0005u, "VLshlrevB64")]
    [InlineData(0xD7000005u, "VLshrrevB64")]
    [InlineData(0xD7010005u, "VAshrrevI64")]
    [InlineData(0xD76F0005u, "VLshlOrB32")]
    [InlineData(0xD7720005u, "VOr3B32")]
    [InlineData(0xD5780005u, "VXor3B32")]
    [InlineData(0xD7760005u, "VSubNcI32")]
    [InlineData(0xD77F0005u, "VAddNcI32")]
    public void Gfx10ExtendedIntegerAndBitwiseFamiliesDecode(
        uint word,
        string expectedOpcode)
    {
        var instruction = DecodeFirst(word, 0x02020501u, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal(Gen5Operand.Vector(1), instruction.Sources[0]);
        Assert.Equal(Gen5Operand.Vector(2), instruction.Sources[1]);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Theory]
    [InlineData(0x120A0501u, "VMulI32I24")]
    [InlineData(0x140A0501u, "VMulHiI32I24")]
    public void Gfx10Integer24Vop2FamiliesDecode(
        uint word,
        string expectedOpcode)
    {
        var instruction = DecodeFirst(word, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Fact]
    public void Gfx10Signed24MadVop3DecodesThreeSources()
    {
        var instruction = DecodeFirst(
            0xD5420005u, // v_mad_i32_i24 v5, v1, v2, v3
            0x040E0501u,
            SEndpgm);

        Assert.Equal("VMadI32I24", instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Fact]
    public void Gfx10LerpU8DecodesPackedThreeSourceVop3()
    {
        var instruction = DecodeFirst(
            0xD54D0005u, // v_lerp_u8 v5, v1, v2, v3
            0x040E0501u,
            SEndpgm);

        Assert.Equal("VLerpU8", instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Fact]
    public void Gfx10DivFmasF32DecodesThreeSourceVop3()
    {
        var instruction = DecodeFirst(
            0xD56F0005u, // v_div_fmas_f32 v5, v1, v2, v3
            0x040E0501u,
            SEndpgm);

        Assert.Equal("VDivFmasF32", instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Theory]
    [InlineData(0x0C0A0501u, "VMacLegacyF32")]
    [InlineData(0x0E0A0501u, "VMulLegacyF32")]
    public void Gfx10LegacyFloatVop2FamiliesDecode(
        uint word,
        string expectedOpcode)
    {
        var instruction = DecodeFirst(word, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Theory]
    [InlineData(0xD5060005u, "VMacLegacyF32")]
    [InlineData(0xD5070005u, "VMulLegacyF32")]
    [InlineData(0xD5400005u, "VMadLegacyF32")]
    [InlineData(0xD5500005u, "VMullitF32")]
    public void Gfx10LegacyFloatVop3FamiliesDecode(
        uint word,
        string expectedOpcode)
    {
        var instruction = DecodeFirst(word, 0x02020501u, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2)],
            instruction.Sources.Take(2));
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Fact]
    public void Gfx10MullitF32DecodesDocumentedThreeSourceEncoding()
    {
        var instruction = DecodeFirst(
            0xD5500005u, // v_mullit_f32 v5, v1, v2, v3
            0x040E0501u,
            SEndpgm);

        Assert.Equal("VMullitF32", instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Fact]
    public void Gfx10MsadU8DecodesMaskedThreeSourceVop3()
    {
        var instruction = DecodeFirst(
            0xD5710005u, // v_msad_u8 v5, v1, v2, v3
            0x040E0501u,
            SEndpgm);

        Assert.Equal("VMsadU8", instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Theory]
    [InlineData(0xD5720005u, "VQsadPkU16U8", 5u)]
    [InlineData(0xD5730005u, "VMqsadPkU16U8", 5u)]
    [InlineData(0xD57500FCu, "VMqsadU32U8", 252u)]
    public void Gfx10PackedSadFamiliesDecode(
        uint word,
        string expectedOpcode,
        uint expectedDestination)
    {
        var instruction = DecodeFirst(word, 0x040E0501u, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)],
            instruction.Sources);
        Assert.Equal(
            [Gen5Operand.Vector(expectedDestination)],
            instruction.Destinations);
    }

    [Theory]
    [InlineData(0xD4891805u, 0x02020501u, "VCmpLtI16", 0u, 0u, 3u)]
    [InlineData(0xD4210105u, 0x42020501u, "VCmpLtF64", 1u, 2u, 0u)]
    public void Gfx10ExtendedCompareDecodesSourceModifiers(
        uint word,
        uint extra,
        string expectedOpcode,
        uint expectedAbsoluteMask,
        uint expectedNegateMask,
        uint expectedOperandSelect)
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x100);
        uint[] words = [word, extra, SEndpgm];
        Span<byte> shader = stackalloc byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                shader[(index * sizeof(uint))..],
                words[index]);
        }
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var error),
            error);

        Assert.Equal(expectedOpcode, program.Instructions[0].Opcode);
        var control = Assert.IsType<Gen5Vop3Control>(program.Instructions[0].Control);
        Assert.Equal(expectedAbsoluteMask, control.AbsoluteMask);
        Assert.Equal(expectedNegateMask, control.NegateMask);
        Assert.Equal(expectedOperandSelect, control.OperandSelect);
    }

    [Theory]
    [InlineData(0x7E0AA101u, "VCvtF16U16")]
    [InlineData(0x7E0AA301u, "VCvtF16I16")]
    [InlineData(0x7E0AA501u, "VCvtU16F16")]
    [InlineData(0x7E0AA701u, "VCvtI16F16")]
    [InlineData(0x7E0AA901u, "VRcpF16")]
    [InlineData(0x7E0AAB01u, "VSqrtF16")]
    [InlineData(0x7E0AAD01u, "VRsqF16")]
    [InlineData(0x7E0AAF01u, "VLogF16")]
    [InlineData(0x7E0AB101u, "VExpF16")]
    [InlineData(0x7E0A7301u, "VFfbhU32")]
    [InlineData(0x7E0A7701u, "VFfbhI32")]
    [InlineData(0x7E0A7F01u, "VFrexpExpI32F32")]
    [InlineData(0x7E0A8101u, "VFrexpMantF32")]
    [InlineData(0x7E0A7901u, "VFrexpExpI32F64")]
    [InlineData(0x7E0A7B01u, "VFrexpMantF64")]
    [InlineData(0x7E0A0701u, "VCvtI32F64")]
    [InlineData(0x7E0A0901u, "VCvtF64I32")]
    [InlineData(0x7E0A1F01u, "VCvtF32F64")]
    [InlineData(0x7E0A2101u, "VCvtF64F32")]
    [InlineData(0x7E0A2B01u, "VCvtU32F64")]
    [InlineData(0x7E0A2D01u, "VCvtF64U32")]
    [InlineData(0x7E0A2F01u, "VTruncF64")]
    [InlineData(0x7E0A3101u, "VCeilF64")]
    [InlineData(0x7E0A3301u, "VRndneF64")]
    [InlineData(0x7E0A3501u, "VFloorF64")]
    public void Gfx10Vop1EncodingsDecode(uint word, string expectedOpcode)
    {
        var instruction = DecodeFirst(word, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal([Gen5Operand.Vector(1)], instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
    }

    [Fact]
    public void Gfx10Vop1E64AliasHasOneSourceAndPreservesModifiers()
    {
        // LLVM GFX10 fixture: v_cvt_f16_u16_e64 v5, v1
        var instruction = DecodeFirst(0xD5D00005u, 0x02010101u, SEndpgm);

        Assert.Equal("VCvtF16U16", instruction.Opcode);
        Assert.Equal([Gen5Operand.Vector(1)], instruction.Sources);
        Assert.Equal([Gen5Operand.Vector(5)], instruction.Destinations);
        var control = Assert.IsType<Gen5Vop3Control>(instruction.Control);
        Assert.Equal(0u, control.AbsoluteMask);
        Assert.Equal(0u, control.NegateMask);
        Assert.Equal(0u, control.OperandSelect);
    }

    [Theory]
    [InlineData(0x7E003600u, 0u, "VPipeflush", false)]
    [InlineData(0xD59B0000u, 0x02010080u, "VPipeflush", true)]
    [InlineData(0x7E008200u, 0u, "VClrexcp", false)]
    [InlineData(0xD5C10000u, 0x02010080u, "VClrexcp", true)]
    public void Gfx10Vop1StateInstructionsHaveNoArtificialOperands(
        uint word,
        uint extra,
        string expectedOpcode,
        bool extended)
    {
        var instruction = extended
            ? DecodeFirst(word, extra, SEndpgm)
            : DecodeFirst(word, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Empty(instruction.Sources);
        Assert.Empty(instruction.Destinations);
    }

    [Theory]
    [InlineData(0x7E0ACB01u, "VSwapB32")]
    [InlineData(0x7E0AD101u, "VSwaprelB32")]
    public void Gfx10SwapDecodesBothArchitecturalOutputs(
        uint word,
        string expectedOpcode)
    {
        var instruction = DecodeFirst(word, SEndpgm);

        Assert.Equal(expectedOpcode, instruction.Opcode);
        Assert.Equal(
            [Gen5Operand.Vector(1), Gen5Operand.Vector(5)],
            instruction.Sources);
        Assert.Equal(
            [Gen5Operand.Vector(5), Gen5Operand.Vector(1)],
            instruction.Destinations);
    }

    [Theory]
    [InlineData(0xCC004005u, "VPkMadI16")]
    [InlineData(0xCC014005u, "VPkMulLoU16")]
    [InlineData(0xCC024005u, "VPkAddI16")]
    [InlineData(0xCC034005u, "VPkSubI16")]
    [InlineData(0xCC044005u, "VPkLshlrevB16")]
    [InlineData(0xCC054005u, "VPkLshrrevB16")]
    [InlineData(0xCC064005u, "VPkAshrrevI16")]
    [InlineData(0xCC074005u, "VPkMaxI16")]
    [InlineData(0xCC084005u, "VPkMinI16")]
    [InlineData(0xCC094005u, "VPkMadU16")]
    [InlineData(0xCC0A4005u, "VPkAddU16")]
    [InlineData(0xCC0B4005u, "VPkSubU16")]
    [InlineData(0xCC0C4005u, "VPkMaxU16")]
    [InlineData(0xCC0D4005u, "VPkMinU16")]
    [InlineData(0xCC134005u, "VDot2F32F16")]
    [InlineData(0xCC144005u, "VDot2I32I16")]
    [InlineData(0xCC154005u, "VDot2U32U16")]
    [InlineData(0xCC164005u, "VDot4I32I8")]
    [InlineData(0xCC174005u, "VDot4U32U8")]
    [InlineData(0xCC184005u, "VDot8I32I4")]
    [InlineData(0xCC194005u, "VDot8U32U4")]
    public void Gfx10PackedIntegerEncodingsDecode(uint word, string expectedOpcode)
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x100);
        uint[] words = [word, 0x1C0A0300u, SEndpgm];
        Span<byte> shader = stackalloc byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                shader[(index * sizeof(uint))..],
                words[index]);
        }
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var error),
            error);

        Assert.Equal(expectedOpcode, program.Instructions[0].Opcode);
    }

    [Fact]
    public void Gfx10PackedHighOperandSelectKeepsArchitecturalSourceOrder()
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x100);
        uint[] words =
        [
            0xCC004005u, // src0 high selector in word0[14]
            0x040A0300u, // src1/src2 high selectors clear
            SEndpgm,
        ];
        Span<byte> shader = stackalloc byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                shader[(index * sizeof(uint))..],
                words[index]);
        }
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var error),
            error);

        var control = Assert.IsType<Gen5Vop3pControl>(program.Instructions[0].Control);
        Assert.Equal(1u, control.OpSelHiMask);
    }

    [Theory]
    [InlineData(0xD6008005u, 0x0A020400u, "VInterpP1F32", 0u, 0u, false, false, 1u, true)]
    [InlineData(0xD6010005u, 0x02020441u, "VInterpP2F32", 1u, 1u, false, false, 0u, false)]
    [InlineData(0xD6020005u, 0x02000082u, "VInterpMovF32", 2u, 2u, false, false, 0u, false)]
    [InlineData(0xD7420005u, 0x0202051Fu, "VInterpP1llF16", 31u, 0u, true, false, 0u, false)]
    [InlineData(0xD7430005u, 0x040E0481u, "VInterpP1lvF16", 1u, 2u, false, false, 0u, false)]
    [InlineData(0xD75A4005u, 0x040E05C0u, "VInterpP2F16", 0u, 3u, true, true, 0u, false)]
    public void Gfx10ExtendedInterpolationDecodesEmbeddedAttribute(
        uint word,
        uint extra,
        string expectedOpcode,
        uint expectedAttribute,
        uint expectedChannel,
        bool expectedAttributeWordHigh,
        bool expectedDestinationHigh,
        uint expectedOutputModifier,
        bool expectedClamp)
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x100);
        uint[] words = [word, extra, SEndpgm];
        Span<byte> shader = stackalloc byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                shader[(index * sizeof(uint))..],
                words[index]);
        }
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var error),
            error);

        var instruction = program.Instructions[0];
        Assert.Equal(expectedOpcode, instruction.Opcode);
        var control = Assert.IsType<Gen5InterpolationControl>(instruction.Control);
        Assert.Equal(expectedAttribute, control.Attribute);
        Assert.Equal(expectedChannel, control.Channel);
        Assert.Equal(expectedAttributeWordHigh, control.AttributeWordHigh);
        Assert.Equal(expectedOpcode == "VInterpP2F16", control.HalfPrecisionResult);
        Assert.Equal(expectedDestinationHigh, control.DestinationHigh);
        Assert.Equal(expectedOutputModifier, control.OutputModifier);
        Assert.Equal(expectedClamp, control.Clamp);
    }

    [Fact]
    public void FlatLoadUbyteInfersScalarBaseAndCompiles()
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x4000);
        uint[] words =
        [
            // v_add_co_u32 v1, vcc_lo, s12, v6
            0xD70F6A01,
            0x00020C0C,
            // v_add_co_ci_u32_sdwa v2, vcc_lo, 0, s13, vcc_lo
            0x50041AF9,
            0x86860680,
            // flat_load_ubyte v0, v[1:2]
            0xDC200000,
            0x007D0001,
            SEndpgm,
        ];
        var shader = new byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                shader.AsSpan(index * sizeof(uint)),
                words[index]);
        }
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var decodeError),
            decodeError);

        var instruction = Assert.Single(
            program.Instructions,
            item => item.Opcode == "FlatLoadUbyte");
        var control = Assert.IsType<Gen5GlobalMemoryControl>(
            instruction.Control);
        Assert.True(control.UsesFlatAddress);
        Assert.Equal(1u, control.VectorAddress);
        Assert.Equal(0u, control.VectorData);
        Assert.Equal(12u, control.ScalarAddress);
        Assert.Equal(
            [
                Gen5Operand.Vector(1),
                Gen5Operand.Vector(2),
                Gen5Operand.Scalar(12),
            ],
            instruction.Sources);

        uint[] userData =
        [
            unchecked((uint)ShaderAddress),
            unchecked((uint)(ShaderAddress >> 32)),
        ];
        var state = new Gen5ShaderState(
            program,
            userData,
            null,
            UserDataScalarRegisterBase: 12);
        Assert.True(
            Gen5ShaderScalarEvaluator.TryEvaluate(
                ctx,
                state,
                out var evaluation,
                out var evaluationError),
            evaluationError);

        var binding = Assert.Single(evaluation.GlobalMemoryBindings);
        Assert.Equal(12u, binding.ScalarAddress);
        Assert.Contains(instruction.Pc, binding.InstructionPcs);
        Assert.True(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state,
                evaluation,
                1,
                1,
                1,
                out var compiled,
                out var compileError),
            compileError);
        Assert.Contains(
            (ushort)SpirvOp.ISub,
            ReadSpirvOpcodes(compiled.Spirv));
    }

    private static IReadOnlyList<ushort> ReadSpirvOpcodes(byte[] spirv)
    {
        Assert.Equal(0, spirv.Length % sizeof(uint));
        Assert.True(spirv.Length >= 5 * sizeof(uint));
        Assert.Equal(
            0x07230203u,
            BinaryPrimitives.ReadUInt32LittleEndian(spirv));

        var opcodes = new List<ushort>();
        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var instruction =
                BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var wordCount = checked((int)(instruction >> 16));
            Assert.InRange(
                wordCount,
                1,
                (spirv.Length - offset) / sizeof(uint));
            opcodes.Add((ushort)instruction);
            offset += wordCount * sizeof(uint);
        }

        return opcodes;
    }

    private static Gen5ShaderInstruction DecodeFirst(params uint[] words)
    {
        var memory = new TestCpuMemory(ShaderAddress, 0x100);
        Span<byte> shader = stackalloc byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                shader[(index * sizeof(uint))..],
                words[index]);
        }
        Assert.True(memory.TryWrite(ShaderAddress, shader));

        var ctx = new CpuContext(memory, Generation.Gen5);
        Assert.True(
            Gen5ShaderTranslator.TryDecodeProgram(
                ctx,
                ShaderAddress,
                out var program,
                out var error),
            error);
        return program.Instructions[0];
    }

    private sealed class TestCpuMemory(ulong baseAddress, int size) : ICpuMemory
    {
        private readonly byte[] _storage = new byte[size];

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!TryResolve(virtualAddress, destination.Length, out var offset))
            {
                return false;
            }

            _storage.AsSpan(offset, destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(
            ulong virtualAddress,
            ReadOnlySpan<byte> source)
        {
            if (!TryResolve(virtualAddress, source.Length, out var offset))
            {
                return false;
            }

            source.CopyTo(_storage.AsSpan(offset, source.Length));
            return true;
        }

        private bool TryResolve(
            ulong virtualAddress,
            int length,
            out int offset)
        {
            offset = 0;
            if (virtualAddress < baseAddress)
            {
                return false;
            }

            var relative = virtualAddress - baseAddress;
            if (relative + (ulong)length > (ulong)_storage.Length)
            {
                return false;
            }

            offset = (int)relative;
            return true;
        }
    }
}
