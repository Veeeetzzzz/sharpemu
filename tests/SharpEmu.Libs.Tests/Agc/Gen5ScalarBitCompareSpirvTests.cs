// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class Gen5ScalarBitCompareSpirvTests
{
    private const ulong ShaderAddress = 0x1_0000_0000;

    [Theory]
    [InlineData(0xBF0E0200u)] // s_bitcmp0_b64 s[0:1], s2
    [InlineData(0xBF0F0200u)] // s_bitcmp1_b64 s[0:1], s2
    public void ScalarBitCompare64Translates(uint instruction)
    {
        var memory = new FakeCpuMemory(ShaderAddress, 0x2000);
        var ctx = new CpuContext(memory, Generation.Gen5);
        Gen5ShaderAtomicDecodeTests.WriteProgram(
            memory,
            ShaderAddress,
            [instruction, 0xBF810000u]);
        var shaderRegisters = new Dictionary<uint, uint>
        {
            [Gen5ShaderAtomicDecodeTests.ComputePgmRsrc2Register] = 16u << 1,
        };

        Assert.True(
            Gen5ShaderTranslator.TryCreateState(
                ctx,
                ShaderAddress,
                0,
                shaderRegisters,
                Gen5ShaderAtomicDecodeTests.ComputeUserDataRegister,
                out var state,
                out var error),
            error);
        Assert.True(
            Gen5ShaderScalarEvaluator.TryEvaluate(ctx, state, out var evaluation, out error),
            error);
        Assert.True(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state,
                evaluation,
                1,
                1,
                1,
                out var shader,
                out error),
            error);
        Assert.NotEmpty(shader.Spirv);
    }

    [Fact]
    public void DocumentedScalarArithmeticOperationsTranslate()
    {
        uint[][] programs =
        [
            [
                0x968081FF, 0x80000000, // s_absdiff_i32 s0, INT_MIN, 1
                0xBF810000,             // s_endpgm
            ],
            [
                0x9B0082FF, 0x80000000, // s_mul_hi_i32 s0, INT_MIN, 2
                0xBF810000,             // s_endpgm
            ],
            [
                0x91828100, // s_ashr_i64 s[2:3], s[0:1], 1
                0xBF810000, // s_endpgm
            ],
        ];

        foreach (var words in programs)
        {
            var (_, shader) = Compile(words);
            Assert.NotEmpty(shader.Spirv);
        }
    }

    [Theory]
    [InlineData(0x7D200300u)] // v_cmpx_f_i32 v0, v1
    [InlineData(0x7D2E0300u)] // v_cmpx_t_i32 v0, v1
    [InlineData(0x7DA00300u)] // v_cmpx_f_u32 v0, v1
    [InlineData(0x7DAE0300u)] // v_cmpx_t_u32 v0, v1
    public void ConstantIntegerCmpxOperationsTranslate(uint instruction)
    {
        var (_, shader) = Compile([instruction, 0xBF810000]);
        Assert.NotEmpty(shader.Spirv);
    }

    [Theory]
    [InlineData(0x60040300u)] // v_cvt_pk_u16_u32 v2, v0, v1
    [InlineData(0x62040300u)] // v_cvt_pk_i16_i32 v2, v0, v1
    public void IntegerPack16OperationsTranslate(uint instruction)
    {
        var (_, shader) = Compile([instruction, 0xBF810000]);
        Assert.NotEmpty(shader.Spirv);
    }

    [Fact]
    public void PackedHalfDotAccumulateTranslates()
    {
        var (_, shader) = Compile(
        [
            0x04040300, // v_dot2c_f32_f16 v2, v0, v1
            0xBF810000, // s_endpgm
        ]);
        Assert.NotEmpty(shader.Spirv);
    }

    [Theory]
    [InlineData(0xD15A0003u)] // v_sad_u8 v3, v0, v1, v2
    [InlineData(0xD15B0003u)] // v_sad_hi_u8 v3, v0, v1, v2
    [InlineData(0xD15C0003u)] // v_sad_u16 v3, v0, v1, v2
    [InlineData(0xD15D0003u)] // v_sad_u32 v3, v0, v1, v2
    [InlineData(0xD15A8003u)] // v_sad_u8 v3, v0, v1, v2 clamp
    [InlineData(0xD15B8003u)] // v_sad_hi_u8 v3, v0, v1, v2 clamp
    [InlineData(0xD15C8003u)] // v_sad_u16 v3, v0, v1, v2 clamp
    [InlineData(0xD15D8003u)] // v_sad_u32 v3, v0, v1, v2 clamp
    public void UnsignedSadOperationsTranslate(uint instruction)
    {
        var (_, shader) = Compile(
        [
            instruction, 0x040A0300, // v_sad_* v3, v0, v1, v2
            0xBF810000,              // s_endpgm
        ]);
        Assert.NotEmpty(shader.Spirv);
    }

    [Theory]
    [InlineData(0xD14E0003u)] // v_alignbit_b32 v3, v0, v1, v2
    [InlineData(0xD14F0003u)] // v_alignbyte_b32 v3, v0, v1, v2
    [InlineData(0xD3440003u)] // v_perm_b32 v3, v0, v1, v2
    [InlineData(0xD3450003u)] // v_xad_u32 v3, v0, v1, v2
    public void AlignOperationsTranslate(uint instruction)
    {
        var (_, shader) = Compile(
        [
            instruction, 0x040A0300,
            0xBF810000,
        ]);

        Assert.NotEmpty(shader.Spirv);
    }

    [Theory]
    [InlineData(0xD34B0003u)] // v_fma_f16
    [InlineData(0xD3510003u)] // v_min3_f16
    [InlineData(0xD3520003u)] // v_min3_i16
    [InlineData(0xD3530003u)] // v_min3_u16
    [InlineData(0xD3540003u)] // v_max3_f16
    [InlineData(0xD3550003u)] // v_max3_i16
    [InlineData(0xD3560003u)] // v_max3_u16
    [InlineData(0xD3570003u)] // v_med3_f16
    [InlineData(0xD3580003u)] // v_med3_i16
    [InlineData(0xD3590003u)] // v_med3_u16
    [InlineData(0xD34B7803u)] // v_fma_f16 op_sel:[1,1,1,1]
    public void ScalarHalfVop3OperationsTranslate(uint instruction)
    {
        var (_, shader) = Compile(
        [
            instruction, 0x040A0300,
            0xBF810000,
        ]);

        Assert.NotEmpty(shader.Spirv);
    }

    [Theory]
    [InlineData(0x64060300u, false)] // v_add_f16
    [InlineData(0x66060300u, false)] // v_sub_f16
    [InlineData(0x68060300u, false)] // v_subrev_f16
    [InlineData(0x6A060300u, false)] // v_mul_f16
    [InlineData(0x6C060300u, false)] // v_fmac_f16
    [InlineData(0x6E060300u, true)] // v_fmamk_f16
    [InlineData(0x70060300u, true)] // v_fmaak_f16
    [InlineData(0x72060300u, false)] // v_max_f16
    [InlineData(0x74060300u, false)] // v_min_f16
    [InlineData(0x76060300u, false)] // v_ldexp_f16
    [InlineData(0x78060300u, false)] // v_pk_fmac_f16
    public void NativeHalfVop2OperationsTranslate(uint instruction, bool hasLiteral)
    {
        var words = hasLiteral
            ? new[] { instruction, 0x3C003C00u, 0xBF810000u }
            : new[] { instruction, 0xBF810000u };
        var (_, shader) = Compile(words);

        Assert.NotEmpty(shader.Spirv);
    }

    [Fact]
    public void ScalarEvaluatorUsesWrappingAbsdiffAndSignedWideOperations()
    {
        var (absdiff, _) = Compile(
        [
            0x968081FF, 0x80000000, // s_absdiff_i32 s0, INT_MIN, 1
            0xBF810000,
        ]);
        Assert.Equal(0x7FFF_FFFFu, absdiff.ScalarRegisters[0]);

        var (mulHi, _) = Compile(
        [
            0x9B0082FF, 0x80000000, // s_mul_hi_i32 s0, INT_MIN, 2
            0xBF810000,
        ]);
        Assert.Equal(0xFFFF_FFFFu, mulHi.ScalarRegisters[0]);

        var (ashr, _) = Compile(
            [
                0x91828100, // s_ashr_i64 s[2:3], s[0:1], 1
                0xBF810000,
            ],
            new Dictionary<uint, uint>
            {
                [0] = 0,
                [1] = 0x8000_0000,
            });
        Assert.Equal(0u, ashr.ScalarRegisters[2]);
        Assert.Equal(0xC000_0000u, ashr.ScalarRegisters[3]);
    }

    private static (Gen5ShaderEvaluation Evaluation, Gen5SpirvShader Shader) Compile(
        uint[] words,
        IReadOnlyDictionary<uint, uint>? userData = null)
    {
        var memory = new FakeCpuMemory(ShaderAddress, 0x2000);
        var ctx = new CpuContext(memory, Generation.Gen5);
        Gen5ShaderAtomicDecodeTests.WriteProgram(memory, ShaderAddress, words);
        var shaderRegisters = new Dictionary<uint, uint>
        {
            [Gen5ShaderAtomicDecodeTests.ComputePgmRsrc2Register] = 16u << 1,
        };
        if (userData is not null)
        {
            foreach (var (register, value) in userData)
            {
                shaderRegisters[Gen5ShaderAtomicDecodeTests.ComputeUserDataRegister + register] = value;
            }
        }

        Assert.True(
            Gen5ShaderTranslator.TryCreateState(
                ctx,
                ShaderAddress,
                0,
                shaderRegisters,
                Gen5ShaderAtomicDecodeTests.ComputeUserDataRegister,
                out var state,
                out var error),
            error);
        Assert.True(
            Gen5ShaderScalarEvaluator.TryEvaluate(ctx, state, out var evaluation, out error),
            error);
        Assert.True(
            Gen5SpirvTranslator.TryCompileComputeShader(
                state,
                evaluation,
                1,
                1,
                1,
                out var shader,
                out error),
            error);
        return (evaluation, shader);
    }
}
