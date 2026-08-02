// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Structural checks over the emitted MSL — these run on every platform because
/// translation is pure text generation; only the runtime tests need a Metal device.
/// </summary>
public sealed class MslTranslationTests
{
    [Fact]
    public void EveryFixtureTranslates()
    {
        foreach (var fixture in Gen5ComputeFixtures.All)
        {
            var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);
            Assert.Equal(Gen5MslStage.Compute, shader.Stage);
            Assert.Equal("gen5_cs", shader.EntryPoint);
            Assert.Contains("kernel void gen5_cs(", shader.Source, StringComparison.Ordinal);
            Assert.Contains("while (active)", shader.Source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ExecMaskedStoresAreGuarded()
    {
        var shader = Gen5ComputeFixtures.CompileOrThrow(Gen5ComputeFixtures.ExecStore);

        // Every buffer store must sit behind the per-lane EXEC guard.
        Assert.Contains("if (exec)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_store_bytes(b0,", shader.Source, StringComparison.Ordinal);

        // s_mov_b32 exec_lo, 0 / -1 must drive the per-lane bool.
        Assert.Contains("exec = ((", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void LoopFixtureProducesMultipleDispatcherBlocks()
    {
        var shader = Gen5ComputeFixtures.CompileOrThrow(Gen5ComputeFixtures.Loop);

        // The backward branch splits the program into at least three blocks and
        // the conditional branch selects between loop head and fallthrough.
        Assert.Contains("case 0u:", shader.Source, StringComparison.Ordinal);
        Assert.Contains("case 1u:", shader.Source, StringComparison.Ordinal);
        Assert.Contains("case 2u:", shader.Source, StringComparison.Ordinal);
        Assert.Contains("pc = (scc) ?", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xBF0E0200u, "scc = !(")]
    [InlineData(0xBF0F0200u, "scc = (")]
    public void ScalarBitCompare64TestsTheSelectedBit(uint instruction, string expectedAssignment)
    {
        var fixture = new Gen5ComputeFixture(
            "scalar-bitcmp64",
            [
                instruction, // s_bitcmp{0,1}_b64 s[0:1], s2
                0xBF810000,   // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("& 63ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("& 1ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("!= 0ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains(expectedAssignment, shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void DocumentedScalarArithmeticOperationsTranslate()
    {
        var cases = new[]
        {
            new Gen5ComputeFixture(
                "scalar-absdiff-i32",
                [
                    0x968081FF, 0x80000000, // s_absdiff_i32 s0, INT_MIN, 1
                    0xBF810000,             // s_endpgm
                ],
                StoreScalarResourceBase: 0,
                StoreBackingBytes: 0),
            new Gen5ComputeFixture(
                "scalar-mul-hi-i32",
                [
                    0x9B0082FF, 0x80000000, // s_mul_hi_i32 s0, INT_MIN, 2
                    0xBF810000,             // s_endpgm
                ],
                StoreScalarResourceBase: 0,
                StoreBackingBytes: 0),
            new Gen5ComputeFixture(
                "scalar-ashr-i64",
                [
                    0x91828100, // s_ashr_i64 s[2:3], s[0:1], 1
                    0xBF810000, // s_endpgm
                ],
                StoreScalarResourceBase: 0,
                StoreBackingBytes: 0),
        };

        foreach (var fixture in cases)
        {
            var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);
            Assert.Contains("kernel void gen5_cs(", shader.Source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Gfx10FrexpVop1FamiliesTranslateWithPairAwareF64Stores()
    {
        var fixture = new Gen5ComputeFixture(
            "frexp-vop1",
            [
                0x7E0A7F01, // v_frexp_exp_i32_f32 v5, v1
                0x7E0A8101, // v_frexp_mant_f32 v5, v1
                0x7E0A7901, // v_frexp_exp_i32_f64 v5, v[1:2]
                0x7E0A7B01, // v_frexp_mant_f64 v[5:6], v[1:2]
                0xBF810000, // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);
        Assert.Contains("clz(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x3FE0000000000000ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v[6]", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void Gfx10F64ConversionVop1FamiliesTranslateWithPairAwareStores()
    {
        var fixture = new Gen5ComputeFixture(
            "f64-conversions-vop1",
            [
                0x7E0A0701, // v_cvt_i32_f64 v5, v[1:2]
                0x7E0A0901, // v_cvt_f64_i32 v[5:6], v1
                0x7E0A1F01, // v_cvt_f32_f64 v5, v[1:2]
                0x7E0A2101, // v_cvt_f64_f32 v[5:6], v1
                0x7E0A2B01, // v_cvt_u32_f64 v5, v[1:2]
                0x7E0A2D01, // v_cvt_f64_u32 v[5:6], v1
                0xBF810000, // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);
        Assert.Contains("0x0010000000000000ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x7FF0000000000000ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v[6]", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void Gfx10F64RoundVop1FamiliesTranslateWithPairAwareStores()
    {
        var fixture = new Gen5ComputeFixture(
            "f64-round-vop1",
            [
                0x7E0A2F01, // v_trunc_f64 v[5:6], v[1:2]
                0x7E0A3101, // v_ceil_f64 v[5:6], v[1:2]
                0x7E0A3301, // v_rndne_f64 v[5:6], v[1:2]
                0x7E0A3501, // v_floor_f64 v[5:6], v[1:2]
                0xBF810000, // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);
        Assert.Contains("0x3FF0000000000000ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x000FFFFFFFFFFFFFul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v[6]", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0x7D200300u)] // v_cmpx_f_i32 v0, v1
    [InlineData(0x7D2E0300u)] // v_cmpx_t_i32 v0, v1
    [InlineData(0x7DA00300u)] // v_cmpx_f_u32 v0, v1
    [InlineData(0x7DAE0300u)] // v_cmpx_t_u32 v0, v1
    public void ConstantIntegerCmpxOperationsTranslate(uint instruction)
    {
        var fixture = new Gen5ComputeFixture(
            "constant-integer-cmpx",
            [instruction, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("exec =", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void DppBankMaskSelectsFourLaneBanks()
    {
        var fixture = new Gen5ComputeFixture(
            "dpp-bank-mask",
            [
                0x7E0002FAu, // v_mov_b32_dpp v0, v0 quad_perm:[0,0,0,0]
                0x11000000u, // row_mask:1 bank_mask:1
                0xBF810000u,
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(
            "(sharpemu_lane >> 2u) & 3u",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD8800000u, 0x03000100u, "atomic_fetch_add_explicit")] // DS_ADD_RTN_U32
    [InlineData(0xD8940000u, 0x03000100u, "atomic_fetch_min_explicit")] // DS_MIN_RTN_I32
    [InlineData(0xD8B40000u, 0x03000100u, "atomic_exchange_explicit")] // DS_WRXCHG_RTN_B32
    [InlineData(0xD8C00000u, 0x03020100u, "atomic_compare_exchange_weak_explicit")] // DS_CMPST_RTN_B32
    [InlineData(0xD8080000u, 0x00000100u, "atomic_compare_exchange_weak_explicit")] // DS_RSUB_U32
    [InlineData(0xD8300000u, 0x00020100u, "atomic_compare_exchange_weak_explicit")] // DS_MSKOR_B32
    [InlineData(0xD8B00000u, 0x03020100u, "atomic_compare_exchange_weak_explicit")] // DS_MSKOR_RTN_B32
    public void DataShareAtomicFamilyTranslates(
        uint instruction,
        uint operands,
        string expectedOperation)
    {
        var fixture = new Gen5ComputeFixture(
            "data-share-atomic",
            [instruction, operands, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedOperation, shader.Source, StringComparison.Ordinal);
        Assert.Contains("if (exec)", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD80C0000u)] // DS_INC_U32 v0, v1
    [InlineData(0xD8900000u)] // DS_DEC_RTN_U32 v3, v0, v1
    public void BoundedDataShareAtomicsUseDocumentedCompareExchangeLoop(uint instruction)
    {
        var fixture = new Gen5ComputeFixture(
            "bounded-data-share-atomic",
            [instruction, 0x03000100u, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("atomic_load_explicit", shader.Source, StringComparison.Ordinal);
        Assert.Contains("atomic_compare_exchange_weak_explicit", shader.Source, StringComparison.Ordinal);
        Assert.Contains("while (true)", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0x1Eu, 0x00000100u, "atomic_compare_exchange_weak_explicit")] // DS_WRITE_B8
    [InlineData(0x1Fu, 0x00000100u, "atomic_compare_exchange_weak_explicit")] // DS_WRITE_B16
    [InlineData(0x39u, 0x02000000u, "uint(int(char(")] // DS_READ_I8
    [InlineData(0x3Au, 0x02000000u, "& 255u")] // DS_READ_U8
    [InlineData(0x3Bu, 0x02000000u, "uint(int(short(")] // DS_READ_I16
    [InlineData(0x3Cu, 0x02000000u, "<< 8u")] // DS_READ_U16
    [InlineData(0x4Du, 0x00000100u, "sharpemu_lds")] // DS_WRITE_B64
    [InlineData(0x4Eu, 0x00030100u, "sharpemu_lds")] // DS_WRITE2_B64
    [InlineData(0x4Fu, 0x00030100u, "sharpemu_lds")] // DS_WRITE2ST64_B64
    [InlineData(0x76u, 0x02000000u, "sharpemu_lds")] // DS_READ_B64
    [InlineData(0x77u, 0x04000000u, "sharpemu_lds")] // DS_READ2_B64
    [InlineData(0x78u, 0x04000000u, "sharpemu_lds")] // DS_READ2ST64_B64
    [InlineData(0xA0u, 0x00000100u, ">> 16u")] // DS_WRITE_B8_D16_HI
    [InlineData(0xA1u, 0x00000100u, ">> 24u")] // DS_WRITE_B16_D16_HI
    [InlineData(0xA2u, 0x02000000u, "v[2] & 0xffff0000u")] // DS_READ_U8_D16
    [InlineData(0xA3u, 0x02000000u, "v[2] & 0x0000ffffu")] // DS_READ_U8_D16_HI
    [InlineData(0xA4u, 0x02000000u, "ushort(short(char(")] // DS_READ_I8_D16
    [InlineData(0xA5u, 0x02000000u, "ushort(short(char(")] // DS_READ_I8_D16_HI
    [InlineData(0xA6u, 0x02000000u, "<< 8u")] // DS_READ_U16_D16
    [InlineData(0xA7u, 0x02000000u, "<< 16u")] // DS_READ_U16_D16_HI
    [InlineData(0xB0u, 0x00000100u, "sharpemu_lane * 4u")] // DS_WRITE_ADDTID_B32
    [InlineData(0xB1u, 0x02000000u, "sharpemu_lane * 4u")] // DS_READ_ADDTID_B32
    [InlineData(0x14u, 0x00000000u, "sharpemu_lds")] // DS_NOP (module still declares LDS)
    public void CommonDataShareLoadsAndStoresTranslate(
        uint opcode,
        uint operands,
        string expectedSource)
    {
        var instruction = 0xD8000000u | (opcode << 18) | 3u;
        var fixture = new Gen5ComputeFixture(
            "data-share-load-store",
            [instruction, operands, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedSource, shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void BufferAtomicFamilyTranslates()
    {
        var cases = new (uint Opcode, string ExpectedOperation)[]
        {
            (0x30, "atomic_exchange_explicit"),
            (0x31, "atomic_compare_exchange_weak_explicit"),
            (0x32, "atomic_fetch_add_explicit"),
            (0x33, "atomic_fetch_sub_explicit"),
            (0x35, "atomic_fetch_min_explicit"),
            (0x36, "atomic_fetch_min_explicit"),
            (0x37, "atomic_fetch_max_explicit"),
            (0x38, "atomic_fetch_max_explicit"),
            (0x39, "atomic_fetch_and_explicit"),
            (0x3A, "atomic_fetch_or_explicit"),
            (0x3B, "atomic_fetch_xor_explicit"),
            (0x3C, "atomic_compare_exchange_weak_explicit"),
            (0x3D, "atomic_compare_exchange_weak_explicit"),
        };

        foreach (var (opcode, expectedOperation) in cases)
        {
            const uint mubufTemplate = 0xE0E04008u;
            var instruction =
                (mubufTemplate & ~(0x7Fu << 18)) |
                (opcode << 18);
            var fixture = new Gen5ComputeFixture(
                $"buffer-atomic-{opcode:X2}",
                [instruction, 0x80000100u, 0xBF810000],
                StoreScalarResourceBase: 0,
                StoreBackingBytes: 64);

            var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

            Assert.Contains(expectedOperation, shader.Source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void GlobalIntegerAtomicFamilyTranslates()
    {
        var cases = new (uint Opcode, string ExpectedOperation)[]
        {
            (0x30, "atomic_exchange_explicit"),
            (0x31, "atomic_compare_exchange_weak_explicit"),
            (0x32, "atomic_fetch_add_explicit"),
            (0x33, "atomic_fetch_sub_explicit"),
            (0x34, "atomic_compare_exchange_weak_explicit"),
            (0x35, "atomic_fetch_min_explicit"),
            (0x36, "atomic_fetch_min_explicit"),
            (0x37, "atomic_fetch_max_explicit"),
            (0x38, "atomic_fetch_max_explicit"),
            (0x39, "atomic_fetch_and_explicit"),
            (0x3A, "atomic_fetch_or_explicit"),
            (0x3B, "atomic_fetch_xor_explicit"),
            (0x3C, "atomic_compare_exchange_weak_explicit"),
            (0x3D, "atomic_compare_exchange_weak_explicit"),
        };

        foreach (var (opcode, expectedOperation) in cases)
        {
            const uint globalTemplate = 0xDC018000u;
            var fixture = new Gen5ComputeFixture(
                $"global-atomic-{opcode:X2}",
                [globalTemplate | (opcode << 18), 0x00000100u, 0xBF810000u],
                StoreScalarResourceBase: 0,
                StoreBackingBytes: 64);

            var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

            Assert.Contains(expectedOperation, shader.Source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void FlatAddressIsNormalizedToInferredBindingBase()
    {
        var fixture = new Gen5ComputeFixture(
            "flat-address",
            [
                0xD70F6A01u, 0x00020C0Cu, // v_add_co_u32 v1, vcc_lo, s12, v6
                0x50041AF9u, 0x86860680u, // v_add_co_ci_u32_sdwa v2, vcc_lo, 0, s13, vcc_lo
                0xDC200000u, 0x007D0001u, // flat_load_ubyte v0, v[1:2]
                0xBF810000u,
            ],
            StoreScalarResourceBase: 12,
            StoreBackingBytes: 64);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("v[1] - s[12]", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ImageAtomicFamilyUsesMetalTextureCompareExchange()
    {
        uint[] opcodes =
        [
            0x0F, 0x10, 0x11, 0x12,
            0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1A, 0x1B, 0x1C,
        ];

        foreach (var opcode in opcodes)
        {
            var shader = CompileImageAtomicOrThrow(opcode, unifiedFormat: 20);

            Assert.Contains(
                "texture2d<uint, access::read_write>",
                shader.Source,
                StringComparison.Ordinal);
            Assert.Contains(".atomic_load(", shader.Source, StringComparison.Ordinal);
            Assert.Contains(
                ".atomic_compare_exchange_weak(",
                shader.Source,
                StringComparison.Ordinal);
            Assert.Contains("if (exec)", shader.Source, StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData(0x1Bu, ">=")] // IMAGE_ATOMIC_INC
    [InlineData(0x1Cu, "== 0u ||")] // IMAGE_ATOMIC_DEC
    public void BoundedImageAtomicsUseDocumentedLimits(uint opcode, string condition)
    {
        var shader = CompileImageAtomicOrThrow(opcode, unifiedFormat: 20);

        Assert.Contains(condition, shader.Source, StringComparison.Ordinal);
        Assert.Contains(".atomic_compare_exchange_weak(", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void SignedImageAtomicUsesR32SintTexture()
    {
        var shader = CompileImageAtomicOrThrow(0x14, unifiedFormat: 21);

        Assert.Contains(
            "texture2d<int, access::read_write>",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains("as_type<int>", shader.Source, StringComparison.Ordinal);
    }

    private static Gen5MslShader CompileImageAtomicOrThrow(
        uint opcode,
        uint unifiedFormat)
    {
        const uint mimgTemplate = 0xF0442100u;
        var instruction =
            (mimgTemplate & ~(0x7Fu << 18)) |
            (opcode << 18);
        var fixture = new Gen5ComputeFixture(
            $"image-atomic-{opcode:X2}",
            [instruction, 0x00010200u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var decoded = program.Instructions[0];
        var control = Assert.IsType<Gen5ImageControl>(decoded.Control);
        var descriptor = new uint[8];
        descriptor[1] = unifiedFormat << 20;
        var binding = new Gen5ImageBinding(
            decoded.Pc,
            decoded.Opcode,
            control,
            descriptor,
            Array.Empty<uint>(),
            MipLevel: 0);
        var state = new Gen5ShaderState(program, new uint[16], Metadata: null);
        var evaluation = new Gen5ShaderEvaluation(
            new uint[128],
            new uint[128],
            [binding],
            Array.Empty<Gen5GlobalMemoryBinding>());

        Assert.True(
            Gen5MslTranslator.TryCompileComputeShader(
                state,
                evaluation,
                1,
                1,
                1,
                out var shader,
                out var error),
            error);
        return shader!;
    }

    [Theory]
    [InlineData(0x60040300u)] // v_cvt_pk_u16_u32 v2, v0, v1
    [InlineData(0x62040300u)] // v_cvt_pk_i16_i32 v2, v0, v1
    public void IntegerPack16OperationsTranslate(uint instruction)
    {
        var fixture = new Gen5ComputeFixture(
            "integer-pack16",
            [instruction, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("& 0xFFFFu", shader.Source, StringComparison.Ordinal);
        Assert.Contains("<< 16", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void PackedHalfDotAccumulateTranslates()
    {
        var fixture = new Gen5ComputeFixture(
            "packed-half-dot-accumulate",
            [
                0x04040300, // v_dot2c_f32_f16 v2, v0, v1
                0xBF810000, // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("as_type<float>(v[2])", shader.Source, StringComparison.Ordinal);
        Assert.Contains("as_type<half>", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0x7E0A8507u, 1)] // v_movreld_b32 v5, v7
    [InlineData(0x7E0A8707u, 1)] // v_movrels_b32 v5, v7
    [InlineData(0x7E0A8907u, 2)] // v_movrelsd_b32 v5, v7
    public void VectorRelativeMovesUseBoundedM0Indexing(
        uint instruction,
        int expectedM0IndexCount)
    {
        var fixture = new Gen5ComputeFixture(
            "vector-relative-move",
            [instruction, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Equal(
            expectedM0IndexCount,
            CountOccurrences(shader.Source, "+ (s[124])"));
        Assert.Contains("< 256u", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xBE852E07u, 1)] // s_movrels_b32 s5, s7
    [InlineData(0xBE852F07u, 2)] // s_movrels_b64 s[5:6], s[7:8]
    [InlineData(0xBE853007u, 1)] // s_movreld_b32 s5, s7
    [InlineData(0xBE853107u, 2)] // s_movreld_b64 s[5:6], s[7:8]
    public void ScalarRelativeMovesDecodeAndUseBoundedM0Indexing(
        uint instruction,
        int expectedBoundsChecks)
    {
        var fixture = new Gen5ComputeFixture(
            "scalar-relative-move",
            [instruction, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("s[124]", shader.Source, StringComparison.Ordinal);
        Assert.Equal(
            expectedBoundsChecks,
            CountOccurrences(shader.Source, "< 128u"));
    }

    [Theory]
    [InlineData(0xBFA10000u)] // s_clause 0
    [InlineData(0xBFA3FFFFu)] // s_waitcnt_depctr 0xffff
    public void SchedulingOnlySoppInstructionsTranslateAsNoOps(uint instruction)
    {
        var fixture = new Gen5ComputeFixture(
            "scheduling-no-op",
            [instruction, 0xBF810000],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("kernel void gen5_cs(", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD15A0003u, "0xFFu", false)] // v_sad_u8
    [InlineData(0xD15B0003u, "<< 16", false)] // v_sad_hi_u8
    [InlineData(0xD15C0003u, "0xFFFFu", false)] // v_sad_u16
    [InlineData(0xD15D0003u, "max(", false)] // v_sad_u32
    [InlineData(0xD15A8003u, "0xFFu", true)] // v_sad_u8 clamp
    [InlineData(0xD15B8003u, "<< 16", true)] // v_sad_hi_u8 clamp
    [InlineData(0xD15C8003u, "0xFFFFu", true)] // v_sad_u16 clamp
    [InlineData(0xD15D8003u, "max(", true)] // v_sad_u32 clamp
    public void UnsignedSadOperationsTranslate(
        uint instruction,
        string operationFragment,
        bool clamp)
    {
        var fixture = new Gen5ComputeFixture(
            "unsigned-sad",
            [
                instruction, 0x040A0300, // v_sad_* v3, v0, v1, v2
                0xBF810000,              // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(operationFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains("max(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("min(", shader.Source, StringComparison.Ordinal);
        Assert.Equal(
            clamp,
            shader.Source.Contains("? 0xFFFFFFFFu", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(0xD14E0003u, ">> ulong((v[2]) & 31u)")] // v_alignbit_b32
    [InlineData(0xD14F0003u, ">= 8u ? 0u")] // v_alignbyte_b32
    [InlineData(0xD3440003u, "sharpemu_perm_byte(v[0], v[1]")] // v_perm_b32
    [InlineData(0xD3450003u, "(v[0]) ^ (v[1])")] // v_xad_u32
    public void AlignOperationsTranslateDocumentedConcatenation(
        uint instruction,
        string expectedSource)
    {
        var fixture = new Gen5ComputeFixture(
            "align",
            [instruction, 0x040A0300u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        if (instruction is 0xD14E0003u or 0xD14F0003u)
        {
            Assert.Contains("ulong(v[0]) << 32ul", shader.Source, StringComparison.Ordinal);
            Assert.Contains("ulong(v[1])", shader.Source, StringComparison.Ordinal);
        }

        Assert.Contains(expectedSource, shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD34B0003u, "fma(")] // v_fma_f16
    [InlineData(0xD3510003u, "fmin(")] // v_min3_f16
    [InlineData(0xD3520003u, "short")] // v_min3_i16
    [InlineData(0xD3530003u, "ushort")] // v_min3_u16
    [InlineData(0xD3540003u, "fmax(")] // v_max3_f16
    [InlineData(0xD3550003u, "short")] // v_max3_i16
    [InlineData(0xD3560003u, "ushort")] // v_max3_u16
    [InlineData(0xD3570003u, "isnan(")] // v_med3_f16
    [InlineData(0xD3580003u, "short")] // v_med3_i16
    [InlineData(0xD3590003u, "ushort")] // v_med3_u16
    [InlineData(0xD34B7803u, "<< 16")] // v_fma_f16 op_sel:[1,1,1,1]
    public void ScalarHalfVop3OperationsPreserveSiblingDestinationHalf(
        uint instruction,
        string expectedFragment)
    {
        var fixture = new Gen5ComputeFixture(
            "scalar-half-vop3",
            [instruction, 0x040A0300u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains("0xFFFF", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xCC0E4003u, 0x1C0A0300u, "fma(")] // v_pk_fma_f16
    [InlineData(0xCC0F4003u, 0x18020300u, " + ")] // v_pk_add_f16
    [InlineData(0xCC104003u, 0x18020300u, " * ")] // v_pk_mul_f16
    [InlineData(0xCC114003u, 0x18020300u, "fmin(")] // v_pk_min_f16
    [InlineData(0xCC124003u, 0x18020300u, "fmax(")] // v_pk_max_f16
    [InlineData(0xCC204003u, 0x1C0A0300u, "as_type<uint>")] // v_fma_mix_f32
    [InlineData(0xCC214003u, 0x1C0A0300u, "0xFFFF0000u")] // v_fma_mixlo_f16
    [InlineData(0xCC224003u, 0x1C0A0300u, "<< 16")] // v_fma_mixhi_f16
    public void PackedHalfAndMixOperationsTranslate(
        uint instruction,
        uint operands,
        string expectedFragment)
    {
        var fixture = new Gen5ComputeFixture(
            "packed-half",
            [instruction, operands, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains("half", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0x64060300u, false, " + ")] // v_add_f16
    [InlineData(0x66060300u, false, " - ")] // v_sub_f16
    [InlineData(0x68060300u, false, " - ")] // v_subrev_f16
    [InlineData(0x6A060300u, false, " * ")] // v_mul_f16
    [InlineData(0x6C060300u, false, "fma(")] // v_fmac_f16
    [InlineData(0x6E060300u, true, "fma(")] // v_fmamk_f16
    [InlineData(0x70060300u, true, "fma(")] // v_fmaak_f16
    [InlineData(0x72060300u, false, "fmax(")] // v_max_f16
    [InlineData(0x74060300u, false, "fmin(")] // v_min_f16
    [InlineData(0x76060300u, false, "ldexp(")] // v_ldexp_f16
    [InlineData(0x78060300u, false, "<< 16")] // v_pk_fmac_f16
    public void NativeHalfVop2OperationsTranslate(
        uint instruction,
        bool hasLiteral,
        string expectedFragment)
    {
        var words = hasLiteral
            ? new[] { instruction, 0x3C003C00u, 0xBF810000u }
            : new[] { instruction, 0xBF810000u };
        var fixture = new Gen5ComputeFixture(
            "native-half-vop2",
            words,
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains("half", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0x7D920501u, "as_type<half>", " < ")]
    [InlineData(0x7D120501u, "(short)", " < ")]
    [InlineData(0x7D520501u, "(ushort)", " < ")]
    [InlineData(0x7DB20501u, "as_type<half>", "exec =")]
    [InlineData(0x7C420501u, "0x8000000000000000ul", " < ")]
    [InlineData(0x7C620501u, "0x8000000000000000ul", "exec =")]
    [InlineData(0x7D420501u, "as_type<long>", " < ")]
    [InlineData(0x7DC20501u, "ulong", " < ")]
    [InlineData(0x7DE804C1u, "ulong", "exec =")]
    public void Gfx10SixteenBitCompareFamiliesTranslate(
        uint instruction,
        string expectedType,
        string expectedOperation)
    {
        var fixture = new Gen5ComputeFixture(
            "sixteen-bit-compare",
            [
                instruction,
                0xBF810000,
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedType, shader.Source, StringComparison.Ordinal);
        Assert.Contains(expectedOperation, shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD6FF0005u, 0x02020501u, "ulong", "<<")]
    [InlineData(0xD7000005u, 0x02020501u, "ulong", ">>")]
    [InlineData(0xD7010005u, 0x02020501u, "as_type<long>", ">>")]
    [InlineData(0xD76F0005u, 0x040A0300u, "<<", "|")]
    [InlineData(0xD7720005u, 0x040A0300u, "|", "v[")]
    [InlineData(0xD5780005u, 0x040A0300u, "^", "v[")]
    [InlineData(0xD7760005u, 0x02020501u, " - ", "v[2]")]
    public void Gfx10ExtendedIntegerAndBitwiseFamiliesTranslate(
        uint instruction,
        uint extra,
        string expectedFragment,
        string secondFragment)
    {
        var fixture = new Gen5ComputeFixture(
            "extended-integer-bitwise",
            [instruction, extra, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains(secondFragment, shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0x120A0501u, 0u, "long(", "uint")]
    [InlineData(0x140A0501u, 0u, "long(", ">> 32")]
    [InlineData(0xD5420005u, 0x040E0501u, "long(", "+ long(")]
    public void Gfx10Signed24FamiliesTranslate(
        uint instruction,
        uint extra,
        string expectedFragment,
        string secondFragment)
    {
        var words = extra == 0
            ? new[] { instruction, 0xBF810000u }
            : new[] { instruction, extra, 0xBF810000u };
        var fixture = new Gen5ComputeFixture(
            "signed-24-families",
            words,
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains(secondFragment, shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void Gfx10LerpU8TranslatesPackedRoundModeAverages()
    {
        var fixture = new Gen5ComputeFixture(
            "lerp-u8",
            [0xD54D0005u, 0x040E0501u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(">> 1u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0xFFu", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void Gfx10MsadU8TranslatesMaskedByteAbsDifferences()
    {
        var fixture = new Gen5ComputeFixture(
            "msad-u8",
            [0xD5710005u, 0x040E0501u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("== 0u ? 0u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("max(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("min(", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void Gfx10DivFmasF32TranslatesVccScaledFusedResult()
    {
        var fixture = new Gen5ComputeFixture(
            "div-fmas-f32",
            [0xD56F0005u, 0x040E0501u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("fma(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("4294967296.0f", shader.Source, StringComparison.Ordinal);
        Assert.Contains("vcc ?", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("legacy-mul-f32", 0x0E0A0501u, "v[5]")]
    [InlineData("legacy-mac-f32", 0xD5060005u, "v[5]")]
    [InlineData("legacy-mad-f32", 0xD5400005u, "v[5]")]
    [InlineData("mullit-f32", 0xD5500005u, "v[5]")]
    public void Gfx10LegacyFloatFamiliesTranslateDx9ZeroProductRules(
        string name,
        uint firstWord,
        string destination)
    {
        var words = firstWord switch
        {
            0xD5060005u or 0xD5400005u => new[] { firstWord, 0x02020501u, 0xBF810000u },
            0xD5500005u => new[] { firstWord, 0x040E0501u, 0xBF810000u },
            _ => new[] { firstWord, 0xBF810000u },
        };
        var shader = Gen5ComputeFixtures.CompileOrThrow(
            new Gen5ComputeFixture(
                name,
                words,
                StoreScalarResourceBase: 0,
                StoreBackingBytes: 0));

        Assert.Contains("0x7FFFFFFFu", shader.Source, StringComparison.Ordinal);
        Assert.Contains("? 0.0f", shader.Source, StringComparison.Ordinal);
        Assert.Contains(destination, shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD5720005u, 0x040E0501u, "v[6]")]
    [InlineData(0xD5730005u, 0x040E0501u, "v[6]")]
    [InlineData(0xD57500FCu, 0x040E0501u, "v[255]")]
    public void Gfx10PackedSadFamiliesTranslatePairOrQuadStores(
        uint instruction,
        uint extra,
        string expectedStore)
    {
        var fixture = new Gen5ComputeFixture(
            "packed-sad",
            [instruction, extra, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedStore, shader.Source, StringComparison.Ordinal);
        Assert.Contains("max(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("min(", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ExtendedInteger16CompareUsesSelectedSourceHalves()
    {
        var fixture = new Gen5ComputeFixture(
            "extended-i16-compare-op-sel",
            [
                0xD4891805u, // v_cmp_lt_i16_e64 s5, v1, v2 op_sel:[1,1]
                0x02020501u,
                0xBF810000u,
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("((v[1]) >> 16)", shader.Source, StringComparison.Ordinal);
        Assert.Contains("((v[2]) >> 16)", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ExtendedFloat64CompareAppliesAbsoluteThenNegateModifiers()
    {
        var fixture = new Gen5ComputeFixture(
            "extended-f64-compare-modifiers",
            [
                0xD4210105u, // v_cmp_lt_f64_e64 s5, abs(v1), -v2
                0x42020501u,
                0xBF810000u,
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(
            "((((ulong)v[1] | ((ulong)v[2] << 32))) & 0x7FFFFFFFFFFFFFFFul",
            shader.Source,
            StringComparison.Ordinal);
        Assert.Contains(
            "((((ulong)v[2] | ((ulong)v[3] << 32))) ^ 0x8000000000000000ul",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xCC004005u, "short", "*")]
    [InlineData(0xCC014005u, "uint", "*")]
    [InlineData(0xCC024005u, "short", "+")]
    [InlineData(0xCC034005u, "short", "-")]
    [InlineData(0xCC044005u, "uint", "<<")]
    [InlineData(0xCC054005u, "uint", ">>")]
    [InlineData(0xCC064005u, "short", ">>")]
    [InlineData(0xCC074005u, "short", "max(")]
    [InlineData(0xCC084005u, "short", "min(")]
    [InlineData(0xCC094005u, "uint", "*")]
    [InlineData(0xCC0A4005u, "uint", "+")]
    [InlineData(0xCC0B4005u, "uint", "-")]
    [InlineData(0xCC0C4005u, "uint", "max(")]
    [InlineData(0xCC0D4005u, "uint", "min(")]
    public void PackedIntegerFamilyTranslates(
        uint instruction,
        string expectedType,
        string expectedOperation)
    {
        var fixture = new Gen5ComputeFixture(
            "packed-integer",
            [instruction, 0x1C0A0300u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedType, shader.Source, StringComparison.Ordinal);
        Assert.Contains(expectedOperation, shader.Source, StringComparison.Ordinal);
        Assert.Contains("0xFFFFu", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void PackedIntegerHighLaneSelectUsesSourceZeroBitFourteen()
    {
        var fixture = new Gen5ComputeFixture(
            "packed-integer-asymmetric-high-select",
            [0xCC004005u, 0x040A0300u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("((v[0]) >> 16)", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("((v[1]) >> 16)", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("((v[2]) >> 16)", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xCC14C005u, "long", "short", 2)]
    [InlineData(0xCC15C005u, "ulong", "0xFFFFu", 2)]
    [InlineData(0xCC16C005u, "long", "char", 4)]
    [InlineData(0xCC17C005u, "ulong", "0xFFu", 4)]
    [InlineData(0xCC18C005u, "long", "<< 28", 8)]
    [InlineData(0xCC19C005u, "ulong", "0xFu", 8)]
    public void PackedIntegerDotFamilyUsesWideSaturatingAccumulator(
        uint instruction,
        string expectedAccumulator,
        string expectedExtraction,
        int expectedProducts)
    {
        var fixture = new Gen5ComputeFixture(
            "packed-integer-dot",
            [instruction, 0x040A0300u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedAccumulator, shader.Source, StringComparison.Ordinal);
        Assert.Contains(expectedExtraction, shader.Source, StringComparison.Ordinal);
        Assert.True(
            shader.Source.Split(" * ", StringSplitOptions.None).Length - 1 >= expectedProducts);
        Assert.Contains(
            expectedAccumulator == "long" ? "2147483647" : "0xFFFFFFFFul",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void PackedFloatDotUsesPackedLaneAndScalarAccumulatorModifiers()
    {
        var fixture = new Gen5ComputeFixture(
            "packed-float-dot",
            [0xCC13CD05u, 0xA40A0300u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.True(
            shader.Source.Split("fma(", StringSplitOptions.None).Length - 1 >= 2);
        Assert.Contains("fabs(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x7F800000u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x007FFFFFu", shader.Source, StringComparison.Ordinal);
        Assert.Contains("< 1.0f", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD7030005u, "+")]
    [InlineData(0xD7040005u, "-")]
    [InlineData(0xD7058005u, "65535u")]
    [InlineData(0xD7070005u, ">>")]
    [InlineData(0xD7080005u, "(short)")]
    [InlineData(0xD7090005u, "max(")]
    [InlineData(0xD70A0005u, "max(")]
    [InlineData(0xD70B0005u, "min(")]
    [InlineData(0xD70C0005u, "min(")]
    [InlineData(0xD70D8005u, "-32768")]
    [InlineData(0xD70E0005u, "-")]
    [InlineData(0xD7147805u, "<< 16")]
    public void DocumentedVop3Integer16OperationsTranslate(
        uint instruction,
        string expectedFragment)
    {
        var fixture = new Gen5ComputeFixture(
            "vop3-integer16",
            [
                instruction,
                0x02020501u,
                0xBF810000u,
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains("0xFFFFu", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD7110005u, "<< 16")]
    [InlineData(0xD7120005u, "pack_float_to_snorm2x16")]
    [InlineData(0xD7130005u, "pack_float_to_unorm2x16")]
    public void DocumentedVop3HalfPackingOperationsTranslate(
        uint instruction,
        string expectedFragment)
    {
        var fixture = new Gen5ComputeFixture(
            "vop3-half-pack",
            [instruction, 0x02020501u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xD7408005u, "65535u")]
    [InlineData(0xD75E8005u, "-32768")]
    [InlineData(0xD7730005u, "(uint)")]
    [InlineData(0xD7750005u, "as_type<int>")]
    public void DocumentedMixedWidthMadOperationsTranslate(
        uint instruction,
        string expectedFragment)
    {
        var fixture = new Gen5ComputeFixture(
            "mixed-width-mad",
            [instruction, 0x040E0501u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(expectedFragment, shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void HalfDivisionFixupTranslatesAllSpecialCases()
    {
        var fixture = new Gen5ComputeFixture(
            "half-div-fixup",
            [0xD75F8005u, 0x040E0501u, 0xBF810000u],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains("0xFE00u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x7C00u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x0200u", shader.Source, StringComparison.Ordinal);
        Assert.Contains("0x8000u", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0x041Fu, "^ 1u")] // bitmask SWAPX1
    [InlineData(0x80E4u, "& 0x1Cu")] // quad identity
    [InlineData(0xC020u, "+ 1u")] // rotate left by one
    [InlineData(0xE01Fu, "reverse_bits(")] // FFT, no-swizzle mask
    public void EveryDsSwizzleModeUsesSimdShuffleWithoutLds(
        uint pattern,
        string modeFragment)
    {
        var fixture = new Gen5ComputeFixture(
            "ds-swizzle",
            [
                0xD8D40000u | pattern, 0x02000100, // ds_swizzle_b32 v2, v1
                0xBF810000,                        // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Contains(modeFragment, shader.Source, StringComparison.Ordinal);
        Assert.Contains("simd_shuffle(", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "threadgroup uint sharpemu_lds[",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0xDAC80004u, 96)] // ds_permute_b32 v2, v0, v1 offset:4
    [InlineData(0xDACC0004u, 2)] // ds_bpermute_b32 v2, v0, v1 offset:4
    public void DsPermutesDecodeAndUseHalfWaveShufflesWithoutLds(
        uint instruction,
        int expectedShuffleCount)
    {
        var fixture = new Gen5ComputeFixture(
            "ds-permute",
            [
                instruction, 0x02000100,
                0xBF810000,
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);

        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);

        Assert.Equal(
            expectedShuffleCount,
            shader.Source.Split("simd_shuffle(", StringSplitOptions.None).Length - 1);
        Assert.Contains("+ 4u) >> 2) & 31u", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "threadgroup uint sharpemu_lds[",
            shader.Source,
            StringComparison.Ordinal);
    }

    [Fact]
    public void DispatcherIsBoundedByDefault()
    {
        var shader = Gen5ComputeFixtures.CompileOrThrow(Gen5ComputeFixtures.Fmac);
        Assert.Contains("if (++steps >=", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void UniformsCarryDispatchLimitAndBufferLengths()
    {
        var shader = Gen5ComputeFixtures.CompileOrThrow(Gen5ComputeFixtures.ExecStore);
        Assert.Contains("struct SharpEmuUniforms", shader.Source, StringComparison.Ordinal);
        Assert.Contains("dispatch_limit_x", shader.Source, StringComparison.Ordinal);
        Assert.Contains("buffer_bytes[", shader.Source, StringComparison.Ordinal);

        // One global binding: b0 at [[buffer(0)]], uniforms at [[buffer(1)]].
        Assert.Contains("device uint* b0 [[buffer(0)]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("[[buffer(1)]]", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void PixelStageEmitsFragmentInterface()
    {
        var shader = Gen5ComputeFixtures.CompilePixelOrThrow();

        Assert.Equal(Gen5MslStage.Pixel, shader.Stage);
        Assert.Equal("gen5_ps", shader.EntryPoint);
        Assert.Equal(1u, shader.AttributeCount);
        Assert.Contains("fragment Gen5PsOut gen5_ps(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 attr0 [[user(locn0)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("[[color(0)]]", shader.Source, StringComparison.Ordinal);
        Assert.Contains("[[position]]", shader.Source, StringComparison.Ordinal);

        // Interpolation reads land in VGPRs; the export writes MRT0 under EXEC
        // and inactive lanes discard at the end.
        Assert.Contains("as_type<uint>(sharpemu_in.attr0[0])", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_out.mrt0 = exec ?", shader.Source, StringComparison.Ordinal);
        Assert.Contains("discard_fragment();", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ExtendedHalfInterpolationPacksSelectedDestinationHalf()
    {
        uint[] words =
        [
            0xD75AC005u, 0x0C0E05C0u, // attr0.w high, dst high, mul:2, clamp
            0xF8001801u, 0x05050505u, // exp mrt0 v5 done vm
            0xBF810000u,
        ];

        var shader = Gen5ComputeFixtures.CompilePixelOrThrow(words);

        Assert.Contains("sharpemu_in.attr0[3]", shader.Source, StringComparison.Ordinal);
        Assert.Contains(" * 2.0f", shader.Source, StringComparison.Ordinal);
        Assert.Contains("clamp(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("as_type<ushort>(half(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v[5] & 0x0000FFFFu", shader.Source, StringComparison.Ordinal);
        Assert.Contains("<< 16", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void PixelOutputKindsSelectTheAttachmentType()
    {
        var uintShader = Gen5ComputeFixtures.CompilePixelOrThrow(Gen5PixelOutputKind.Uint);
        Assert.Contains("uint4 mrt0 [[color(0)]];", uintShader.Source, StringComparison.Ordinal);

        var sintShader = Gen5ComputeFixtures.CompilePixelOrThrow(Gen5PixelOutputKind.Sint);
        Assert.Contains("int4 mrt0 [[color(0)]];", sintShader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void VertexStageEmitsVertexInterface()
    {
        var shader = Gen5ComputeFixtures.CompileVertexOrThrow();

        Assert.Equal(Gen5MslStage.Vertex, shader.Stage);
        Assert.Equal("gen5_vs", shader.EntryPoint);
        Assert.Equal(1u, shader.AttributeCount);
        Assert.Contains("vertex Gen5VsOut gen5_vs(", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 sharpemu_position [[position]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 param0 [[user(locn0)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("uint sharpemu_vertex_id [[vertex_id]],", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v[5] = sharpemu_vertex_id;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v[8] = sharpemu_instance_id;", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_out.sharpemu_position = exec ?", shader.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_out.param0 = exec ?", shader.Source, StringComparison.Ordinal);
        Assert.Contains("return sharpemu_out;", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void RequiredVertexOutputsAreZeroFilledDeclarations()
    {
        // The paired fragment shader reads locations 0..2; the program only
        // exports param0, so 1 and 2 must still be declared (zero-filled).
        var shader = Gen5ComputeFixtures.CompileVertexOrThrow(requiredVertexOutputCount: 3);
        Assert.Equal(3u, shader.AttributeCount);
        Assert.Contains("float4 param1 [[user(locn1)]];", shader.Source, StringComparison.Ordinal);
        Assert.Contains("float4 param2 [[user(locn2)]];", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void FixedShadersCoverThePresenterSurface()
    {
        var fullscreen = MslFixedShaders.CreateFullscreenVertex(2);
        Assert.Contains("vertex FullscreenOut fullscreen_vs(", fullscreen, StringComparison.Ordinal);
        Assert.Contains("float4 attr1 [[user(locn1)]];", fullscreen, StringComparison.Ordinal);

        Assert.Contains("tex0.sample(smp0, in.attr0.xy)", MslFixedShaders.CreateCopyFragment(), StringComparison.Ordinal);
        Assert.Contains("float4(1.0f, 0.0f, 1.0f, 1.0f)", MslFixedShaders.CreateSolidFragment(1f, 0f, 1f, 1f), StringComparison.Ordinal);
        Assert.Contains("return in.attr3;", MslFixedShaders.CreateAttributeFragment(3), StringComparison.Ordinal);
        Assert.Contains("fragment void depth_only_fs()", MslFixedShaders.CreateDepthOnlyFragment(), StringComparison.Ordinal);
    }

    [Fact]
    public void Gfx10F64FractVop1TranslatesWithPairAwareStores()
    {
        var fixture = new Gen5ComputeFixture(
            "unsupported",
            [
                0x7E0A7D01, // v_fract_f64 v[5:6], v[1:2]
                0xBF810000, // s_endpgm
            ],
            StoreScalarResourceBase: 0,
            StoreBackingBytes: 0);
        var shader = Gen5ComputeFixtures.CompileOrThrow(fixture);
        Assert.Contains("0x0020000000000000ul", shader.Source, StringComparison.Ordinal);
        Assert.Contains("v[6]", shader.Source, StringComparison.Ordinal);
    }

    private static int CountOccurrences(string text, string value) =>
        (text.Length - text.Replace(value, string.Empty, StringComparison.Ordinal).Length) /
        value.Length;
}
