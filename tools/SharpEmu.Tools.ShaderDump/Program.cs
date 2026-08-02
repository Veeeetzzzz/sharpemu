// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Synthetic-shader conformance dumper.
//
// Feeds hand-assembled Gen5 (gfx10) instruction words through the real
// decode -> SPIR-V pipeline (SharpEmu.ShaderCompiler + SharpEmu.ShaderCompiler.Vulkan)
// and writes the resulting vertex, pixel, and compute SPIR-V blobs to disk. The blobs
// can then be checked with spirv-val / spirv-dis.
//
// Programs that contain buffer_store_dword automatically get a single
// global-memory binding covering every store, which the emitter exposes as
// guestBuffers[0] (descriptor set 0, binding 0).
//
// Each program carries an expectation: ExpectTranslate=true programs must
// decode and emit the requested stages; ExpectTranslate=false programs pin a decode
// failure that must stay loud. Any unexpected outcome makes the tool exit
// non-zero, so it can gate scripts/CI.
//
// Usage: SharpEmu.Tools.ShaderDump [output-directory]

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Vulkan;

const ulong ProgramAddress = 0x100000;

(string Name, bool ExpectTranslate, uint[] Words)[] testPrograms =
[
    ("fmac", true, [
        0x560A0501,             // v_fmac_f32 v5, v1, v2
        0x580A0501, 0x42280000, // v_fmamk_f32 v5, v1, 42.0, v2
        0x5A0A0501, 0x42280000, // v_fmaak_f32 v5, v1, v2, 42.0
        0xD52B0005, 0x00020501, // v_fmac_f32_e64 v5, v1, v2
        0xBF810000,             // s_endpgm
    ]),
    ("muls", true, [
        0xD5690005, 0x00020501, // v_mul_lo_u32 v5, v1, v2
        0xD56A0005, 0x00020501, // v_mul_hi_u32 v5, v1, v2
        0xD56B0005, 0x00020501, // v_mul_lo_i32 v5, v1, v2
        0xD56C0005, 0x00020501, // v_mul_hi_i32 v5, v1, v2
        0xBF810000,             // s_endpgm
    ]),
    ("vop1-gfx10", true, [
        0x7E0A7301, // v_ffbh_u32 v5, v1
        0x7E0C7701, // v_ffbh_i32 v6, v1
        0x7E0EA101, // v_cvt_f16_u16 v7, v1
        0x7E10A301, // v_cvt_f16_i16 v8, v1
        0x7E12A501, // v_cvt_u16_f16 v9, v1
        0x7E14A701, // v_cvt_i16_f16 v10, v1
        0x7E16A901, // v_rcp_f16 v11, v1
        0x7E18AB01, // v_sqrt_f16 v12, v1
        0x7E1AAD01, // v_rsq_f16 v13, v1
        0x7E1CAF01, // v_log_f16 v14, v1
        0x7E1EB101, // v_exp_f16 v15, v1
        0x7E20B301, // v_frexp_mant_f16 v16, v1
        0x7E22B501, // v_frexp_exp_i16_f16 v17, v1
        0x7E24B701, // v_floor_f16 v18, v1
        0x7E26B901, // v_ceil_f16 v19, v1
        0x7E28BB01, // v_trunc_f16 v20, v1
        0x7E2ABD01, // v_rndne_f16 v21, v1
        0x7E2CBF01, // v_fract_f16 v22, v1
        0x7E2EC101, // v_sin_f16 v23, v1
        0x7E30C301, // v_cos_f16 v24, v1
        0x7E32C501, // v_sat_pk_u8_i16 v25, v1
        0x7E34C701, // v_cvt_norm_i16_f16 v26, v1
        0x7E36C901, // v_cvt_norm_u16_f16 v27, v1
        0x7E003600, // v_pipeflush
        0x7E008200, // v_clrexcp
        0xBF810000, // s_endpgm
    ]),
    ("frexp-vop1", true, [
        0x7E0A7F01, // v_frexp_exp_i32_f32 v5, v1
        0x7E0A8101, // v_frexp_mant_f32 v5, v1
        0x7E0A7901, // v_frexp_exp_i32_f64 v5, v[1:2]
        0x7E0A7B01, // v_frexp_mant_f64 v[5:6], v[1:2]
        0xBF810000, // s_endpgm
    ]),
    ("f64-conversions-vop1", true, [
        0x7E0A0701, // v_cvt_i32_f64 v5, v[1:2]
        0x7E0A0901, // v_cvt_f64_i32 v[5:6], v1
        0x7E0A1F01, // v_cvt_f32_f64 v5, v[1:2]
        0x7E0A2101, // v_cvt_f64_f32 v[5:6], v1
        0x7E0A2B01, // v_cvt_u32_f64 v5, v[1:2]
        0x7E0A2D01, // v_cvt_f64_u32 v[5:6], v1
        0xBF810000, // s_endpgm
    ]),
    ("f64-round-vop1", true, [
        0x7E0A2F01, // v_trunc_f64 v[5:6], v[1:2]
        0x7E0A3101, // v_ceil_f64 v[5:6], v[1:2]
        0x7E0A3301, // v_rndne_f64 v[5:6], v[1:2]
        0x7E0A3501, // v_floor_f64 v[5:6], v[1:2]
        0x7E0A7D01, // v_fract_f64 v[5:6], v[1:2]
        0xBF810000, // s_endpgm
    ]),
    // Packed f16 (VOP3P) arithmetic, including the fused multiply-add. The
    // constants pin the double-rounding regression from the VOP3P first slice:
    // fma(0x4100, 0x7522, 0x04EA) must round once to 0x7A6B (an f32
    // multiply-add then pack yields 0x7A6A). The last fma exercises the src2
    // neg_lo/neg_hi modifier path.
    ("pk-f16", true, [
        0x7E0002FF, 0x41004100, // v_mov_b32 v0, 0x41004100 (2.5 packed)
        0x7E0202FF, 0x75227522, // v_mov_b32 v1, 0x75227522 (21024 packed)
        0x7E0402FF, 0x04EA04EA, // v_mov_b32 v2, 0x04EA04EA (~7.496e-5 packed)
        0xCC0E4003, 0x1C0A0300, // v_pk_fma_f16 v3, v0, v1, v2
        0xCC0F4004, 0x18020500, // v_pk_add_f16 v4, v0, v2
        0xCC104005, 0x18020300, // v_pk_mul_f16 v5, v0, v1
        0xCC114006, 0x18020300, // v_pk_min_f16 v6, v0, v1
        0xCC124007, 0x18020300, // v_pk_max_f16 v7, v0, v1
        0xCC0E4408, 0x9C0A0300, // v_pk_fma_f16 v8, v0, v1, neg_lo:[0,0,1] neg_hi:[0,0,1] v2
        0xCC0FC009, 0x18020000, // v_pk_add_f16 v9, v0, v0 clamp  (2.5+2.5=5 -> saturates to 1.0)
        0xCC0EC40A, 0x1C0A0300, // v_pk_fma_f16 v10, v0, v1, v2 clamp
        0xBF810000,             // s_endpgm
    ]),
    ("mrt", true, [
        0x7E0002FF, 0x3F800000, // v_mov_b32 v0, 1.0f
        0x7E0202FF, 0x00000000, // v_mov_b32 v1, 0.0f
        0x7E0402FF, 0x00000000, // v_mov_b32 v2, 0.0f
        0x7E0602FF, 0x3F800000, // v_mov_b32 v3, 1.0f
        0x7E0802FF, 0x00000001, // v_mov_b32 v4, 1u
        0x7E0A02FF, 0x00000002, // v_mov_b32 v5, 2u
        0x7E0C02FF, 0x00000003, // v_mov_b32 v6, 3u
        0x7E0E02FF, 0x00000004, // v_mov_b32 v7, 4u
        0x7E1002FF, 0xFFFFFFFF, // v_mov_b32 v8, -1
        0x7E1202FF, 0x00000002, // v_mov_b32 v9, 2
        0x7E1402FF, 0xFFFFFFFD, // v_mov_b32 v10, -3
        0x7E1602FF, 0x00000004, // v_mov_b32 v11, 4
        0xF800000F, 0x03020100, // exp mrt0 v0, v1, v2, v3
        0xF800003F, 0x07060504, // exp mrt3 v4, v5, v6, v7
        0xF800086F, 0x0B0A0908, // exp mrt6 v8, v9, v10, v11 done
        0xBF810000,             // s_endpgm
    ]),
    ("mrt-float2", true, [
        0x7E0002FF, 0x3F800000, // v_mov_b32 v0, 1.0f
        0x7E0202FF, 0x3E800000, // v_mov_b32 v1, 0.25f
        0x7E0402FF, 0x3E800000, // v_mov_b32 v2, 0.25f
        0x7E0602FF, 0x3F000000, // v_mov_b32 v3, 0.5f
        0xF800000F, 0x03020100, // exp mrt0 v0, v1, v2, v3
        0xF800081F, 0x03020100, // exp mrt1 v0, v1, v2, v3 done
        0xBF810000,             // s_endpgm
    ]),
    ("mrt8", true, [
        0x7E0002FF, 0x3F800000, // v_mov_b32 v0, 1.0f
        0x7E0202FF, 0x00000000, // v_mov_b32 v1, 0.0f
        0x7E0402FF, 0x00000000, // v_mov_b32 v2, 0.0f
        0x7E0602FF, 0x3F800000, // v_mov_b32 v3, 1.0f
        0xF800000F, 0x03020100, // exp mrt0 v0, v1, v2, v3
        0xF800001F, 0x03020100, // exp mrt1 v0, v1, v2, v3
        0xF800002F, 0x03020100, // exp mrt2 v0, v1, v2, v3
        0xF800003F, 0x03020100, // exp mrt3 v0, v1, v2, v3
        0xF800004F, 0x03020100, // exp mrt4 v0, v1, v2, v3
        0xF800005F, 0x03020100, // exp mrt5 v0, v1, v2, v3
        0xF800006F, 0x03020100, // exp mrt6 v0, v1, v2, v3
        0xF800087F, 0x03020100, // exp mrt7 v0, v1, v2, v3 done
        0xBF810000,             // s_endpgm
    ]),
    ("mrt-partial", true, [
        0x7E0002FF, 0x3F4CCCCD, // v_mov_b32 v0, 0.8f
        0x7E0202FF, 0x3F333333, // v_mov_b32 v1, 0.7f
        0xF8000803, 0x03020100, // exp mrt0 v0, v1, off, off done
        0xBF810000,             // s_endpgm
    ]),
    ("mrt-partial-merge", true, [
        0x7E0002FF, 0x3DCCCCCD, // v_mov_b32 v0, 0.1f
        0x7E0202FF, 0x3E4CCCCD, // v_mov_b32 v1, 0.2f
        0x7E0C02FF, 0x3E99999A, // v_mov_b32 v6, 0.3f
        0x7E0E02FF, 0x3ECCCCCD, // v_mov_b32 v7, 0.4f
        0xF8000003, 0x03020100, // exp mrt0 v0, v1, off, off
        0xF800080C, 0x07060504, // exp mrt0 off, off, v6, v7 done
        0xBF810000,             // s_endpgm
    ]),
    ("sopp-hints", true, [
        0xBFA10001,             // s_clause 0x1
        0xBFA30000,             // s_waitcnt_depctr 0x0
        0xBF810000,             // s_endpgm
    ]),
    ("align", true, [
        0xD14E0003, 0x040A0300, // v_alignbit_b32 v3, v0, v1, v2
        0xD14F0004, 0x040A0300, // v_alignbyte_b32 v4, v0, v1, v2
        0xD3440005, 0x040A0300, // v_perm_b32 v5, v0, v1, v2
        0xD3450006, 0x040A0300, // v_xad_u32 v6, v0, v1, v2
        0xBF810000,             // s_endpgm
    ]),
    ("scalar-half-vop3", true, [
        0xD34B0003, 0x040A0300, // v_fma_f16 v3, v0.lo, v1.lo, v2.lo
        0xD3510004, 0x040A0300, // v_min3_f16 v4, v0.lo, v1.lo, v2.lo
        0xD3520005, 0x040A0300, // v_min3_i16 v5, v0.lo, v1.lo, v2.lo
        0xD3530006, 0x040A0300, // v_min3_u16 v6, v0.lo, v1.lo, v2.lo
        0xD3540007, 0x040A0300, // v_max3_f16 v7, v0.lo, v1.lo, v2.lo
        0xD3550008, 0x040A0300, // v_max3_i16 v8, v0.lo, v1.lo, v2.lo
        0xD3560009, 0x040A0300, // v_max3_u16 v9, v0.lo, v1.lo, v2.lo
        0xD357000A, 0x040A0300, // v_med3_f16 v10, v0.lo, v1.lo, v2.lo
        0xD358000B, 0x040A0300, // v_med3_i16 v11, v0.lo, v1.lo, v2.lo
        0xD359780C, 0x040A0300, // v_med3_u16 v12.hi, v0.hi, v1.hi, v2.hi
        0xBF810000,             // s_endpgm
    ]),
    ("compare-widths", true, [
        0x7D920501,             // v_cmp_lt_f16 vcc, v1, v2
        0x7D120501,             // v_cmp_lt_i16 vcc, v1, v2
        0x7D520501,             // v_cmp_lt_u16 vcc, v1, v2
        0x7C420501,             // v_cmp_lt_f64 vcc, v[1:2], v[2:3]
        0xD4D9007E, 0x02020501, // v_cmpx_lt_f16_e64 v1, v2
        0xBF810000,             // s_endpgm
    ]),
    ("integer16-vop3", true, [
        0xD7030005, 0x02020501, // v_add_nc_u16 v5, v1, v2
        0xD7040005, 0x02020501, // v_sub_nc_u16 v5, v1, v2
        0xD7058005, 0x02020501, // v_mul_lo_u16 v5, v1, v2 clamp
        0xD7070005, 0x02020501, // v_lshrrev_b16 v5, v1, v2
        0xD7080005, 0x02020501, // v_ashrrev_i16 v5, v1, v2
        0xD7090005, 0x02020501, // v_max_u16 v5, v1, v2
        0xD70A0005, 0x02020501, // v_max_i16 v5, v1, v2
        0xD70B0005, 0x02020501, // v_min_u16 v5, v1, v2
        0xD70C0005, 0x02020501, // v_min_i16 v5, v1, v2
        0xD70D8005, 0x02020501, // v_add_nc_i16 v5, v1, v2 clamp
        0xD70E0005, 0x02020501, // v_sub_nc_i16 v5, v1, v2
        0xD7110005, 0x02020501, // v_pack_b32_f16 v5, v1, v2
        0xD7120005, 0x02020501, // v_cvt_pknorm_i16_f16 v5, v1, v2
        0xD7130005, 0x02020501, // v_cvt_pknorm_u16_f16 v5, v1, v2
        0xD7147805, 0x02020501, // v_lshlrev_b16 v5.hi, v1.hi, v2.hi
        0xD7408005, 0x040E0501, // v_mad_u16 v5, v1, v2, v3 clamp
        0xD75E8005, 0x040E0501, // v_mad_i16 v5, v1, v2, v3 clamp
        0xD7730005, 0x040E0501, // v_mad_u32_u16 v5, v1, v2, v3
        0xD7750005, 0x040E0501, // v_mad_i32_i16 v5, v1, v2, v3
        0xD75F8005, 0x040E0501, // v_div_fixup_f16 v5, v1, v2, v3 clamp
        0xBF810000,             // s_endpgm
    ]),
    ("packed-integer-vop3p", true, [
        0xCC004005, 0x1C0A0300, // v_pk_mad_i16 v5, v0, v1, v2
        0xCC014006, 0x1C0A0300, // v_pk_mul_lo_u16 v6, v0, v1
        0xCC024007, 0x1C0A0300, // v_pk_add_i16 v7, v0, v1
        0xCC034008, 0x1C0A0300, // v_pk_sub_i16 v8, v0, v1
        0xCC044009, 0x1C0A0300, // v_pk_lshlrev_b16 v9, v0, v1
        0xCC05400A, 0x1C0A0300, // v_pk_lshrrev_b16 v10, v0, v1
        0xCC06400B, 0x1C0A0300, // v_pk_ashrrev_i16 v11, v0, v1
        0xCC07400C, 0x1C0A0300, // v_pk_max_i16 v12, v0, v1
        0xCC08400D, 0x1C0A0300, // v_pk_min_i16 v13, v0, v1
        0xCC09400E, 0x1C0A0300, // v_pk_mad_u16 v14, v0, v1, v2
        0xCC0A400F, 0x1C0A0300, // v_pk_add_u16 v15, v0, v1
        0xCC0B4010, 0x1C0A0300, // v_pk_sub_u16 v16, v0, v1
        0xCC0C4011, 0x1C0A0300, // v_pk_max_u16 v17, v0, v1
        0xCC0D4012, 0x1C0A0300, // v_pk_min_u16 v18, v0, v1
        0xCC00C013, 0x040A0300, // asymmetric high select + clamp
        0xCC13CD1A, 0xA40A0300, // v_dot2_f32_f16 v26, abs/neg src2, clamp
        0xCC14C014, 0x040A0300, // v_dot2_i32_i16 v20, v0, v1, v2 clamp
        0xCC15C015, 0x040A0300, // v_dot2_u32_u16 v21, v0, v1, v2 clamp
        0xCC16C016, 0x040A0300, // v_dot4_i32_i8 v22, v0, v1, v2 clamp
        0xCC17C017, 0x040A0300, // v_dot4_u32_u8 v23, v0, v1, v2 clamp
        0xCC18C018, 0x040A0300, // v_dot8_i32_i4 v24, v0, v1, v2 clamp
        0xCC19C019, 0x040A0300, // v_dot8_u32_u4 v25, v0, v1, v2 clamp
        0xBF810000,             // s_endpgm
    ]),
    // Common LDS load/store widths. Kept compute-only below because SPIR-V
    // Workgroup storage is not legal in vertex execution models.
    ("lds-common", true, [
        0xD8080003, 0x00000100, // ds_rsub_u32 v0, v1 offset:3
        0xD8300003, 0x00020100, // ds_mskor_b32 v0, v1, v2 offset:3
        0xD8B00003, 0x0F020100, // ds_mskor_rtn_b32 v15, v0, v1, v2 offset:3
        0xD80C0003, 0x00000100, // ds_inc_u32 v0, v1 offset:3
        0xD8900003, 0x0E000100, // ds_dec_rtn_u32 v14, v0, v1 offset:3
        0xD8500000, 0x00000000, // ds_nop
        0xD8780003, 0x00000100, // ds_write_b8 v0, v1 offset:3
        0xD87C0003, 0x00000100, // ds_write_b16 v0, v1 offset:3
        0xD8E40003, 0x10000000, // ds_read_i8 v16, v0 offset:3
        0xD8E80003, 0x11000000, // ds_read_u8 v17, v0 offset:3
        0xD8EC0003, 0x12000000, // ds_read_i16 v18, v0 offset:3
        0xD8F00003, 0x13000000, // ds_read_u16 v19, v0 offset:3
        0xD9340003, 0x00000100, // ds_write_b64 v0, v[1:2] offset:3
        0xD9380703, 0x00030100, // ds_write2_b64 v0, v[1:2], v[3:4] offset0:3 offset1:7
        0xD93C0703, 0x00030100, // ds_write2st64_b64 v0, v[1:2], v[3:4]
        0xD9D80003, 0x14000000, // ds_read_b64 v[20:21], v0 offset:3
        0xD9DC0703, 0x16000000, // ds_read2_b64 v[22:25], v0
        0xD9E00703, 0x1A000000, // ds_read2st64_b64 v[26:29], v0
        0xDA800003, 0x00000100, // ds_write_b8_d16_hi v0, v1 offset:3
        0xDA840003, 0x00000100, // ds_write_b16_d16_hi v0, v1 offset:3
        0xDA880003, 0x1E000000, // ds_read_u8_d16 v30, v0 offset:3
        0xDA8C0003, 0x1F000000, // ds_read_u8_d16_hi v31, v0 offset:3
        0xDA900003, 0x20000000, // ds_read_i8_d16 v32, v0 offset:3
        0xDA940003, 0x21000000, // ds_read_i8_d16_hi v33, v0 offset:3
        0xDA980003, 0x22000000, // ds_read_u16_d16 v34, v0 offset:3
        0xDA9C0003, 0x23000000, // ds_read_u16_d16_hi v35, v0 offset:3
        0xDAC00003, 0x00000100, // ds_write_addtid_b32 v1 offset:3
        0xDAC40003, 0x24000000, // ds_read_addtid_b32 v36 offset:3
        0xBF810000,             // s_endpgm
    ]),
    // s_round_mode / s_denorm_mode write the FP MODE state and must keep
    // failing decode loudly until their semantics are modeled (see #108);
    // this program pins that behavior.
    ("sopp-mode", false, [
        0xBFA40000,             // s_round_mode 0x0
        0xBFA50000,             // s_denorm_mode 0x0
        0xBF810000,             // s_endpgm
    ]),
    // Executable end-to-end test: compute with real ALU instructions, then
    // buffer_store_dword results to guestBuffers[0] at offsets 0/4/8, prove
    // that a store with EXEC=0 does not land (offset 12 stays sentinel), and
    // that stores work again after EXEC is restored (offset 16). Offsets 20/24
    // hold the packed fused f16 FMA and its negated-addend twin, whose exact
    // results (0x7A6B7A6B / 0x7A6A7A6A) straddle an f16 midpoint and therefore
    // catch any double-rounding regression on real hardware.
    ("exec", true, [
        0xBFA10001,             // s_clause 0x1 (hint no-op in an executed program, needs #108)
        0x7E0002FF, 0x3FC00000, // v_mov_b32 v0, 1.5f
        0x7E0202FF, 0x40100000, // v_mov_b32 v1, 2.25f
        0x7E0402FF, 0x41200000, // v_mov_b32 v2, 10.0f
        0x56040300,             // v_fmac_f32 v2, v0, v1      -> v2 = fma(1.5, 2.25, 10.0)
        0x7E0602FF, 0x7FFFFFFF, // v_mov_b32 v3, 0x7FFFFFFF
        0x7E0802FF, 0x00010003, // v_mov_b32 v4, 0x00010003
        0xD56C0005, 0x00020903, // v_mul_hi_i32 v5, v3, v4
        0xD56B0006, 0x00020903, // v_mul_lo_i32 v6, v3, v4
        0xE0700000, 0x80020200, // buffer_store_dword v2, off, s[8:11], 0
        0xE0700004, 0x80020500, // buffer_store_dword v5, off, s[8:11], 0 offset:4
        0xE0700008, 0x80020600, // buffer_store_dword v6, off, s[8:11], 0 offset:8
        0xBEFE0380,             // s_mov_b32 exec_lo, 0       -> lane inactive
        0xE070000C, 0x80020200, // buffer_store_dword v2, off, s[8:11], 0 offset:12 (masked, must not land)
        0xBEFE03C1,             // s_mov_b32 exec_lo, -1      -> lane active again
        0xE0700010, 0x80020000, // buffer_store_dword v0, off, s[8:11], 0 offset:16
        0x7E0E02FF, 0x41004100, // v_mov_b32 v7, 0x41004100 (2.5 packed)
        0x7E1002FF, 0x75227522, // v_mov_b32 v8, 0x75227522 (21024 packed)
        0x7E1202FF, 0x04EA04EA, // v_mov_b32 v9, 0x04EA04EA (~7.496e-5 packed)
        0xCC0E400A, 0x1C261107, // v_pk_fma_f16 v10, v7, v8, v9
        0xCC0E440B, 0x9C261107, // v_pk_fma_f16 v11, v7, v8, neg_lo:[0,0,1] neg_hi:[0,0,1] v9
        0xCC0EC00C, 0x1C261107, // v_pk_fma_f16 v12, v7, v8, v9 clamp (>=1 -> saturates to 1.0)
        0xE0700014, 0x80020A00, // buffer_store_dword v10, off, s[8:11], 0 offset:20
        0xE0700018, 0x80020B00, // buffer_store_dword v11, off, s[8:11], 0 offset:24
        0xE070001C, 0x80020C00, // buffer_store_dword v12, off, s[8:11], 0 offset:28
        0xBF810000,             // s_endpgm
    ]),
];

var outputDirectory = args.Length > 0
    ? args[0]
    : Path.Combine(AppContext.BaseDirectory, "spv");
Directory.CreateDirectory(outputDirectory);

var failures = 0;
foreach (var (name, expectTranslate, words) in testPrograms)
{
    var memory = new FakeMemory();
    memory.AddRegion(ProgramAddress, words);
    var ctx = new CpuContext(memory, Generation.Gen5);

    Console.WriteLine(
        $"[{name}] decode: " +
        Gen5ShaderTranslator.Describe(ctx, ProgramAddress, ProgramAddress));

    if (!Gen5ShaderTranslator.TryDecodeProgram(ctx, ProgramAddress, out var program, out var decodeError))
    {
        if (expectTranslate)
        {
            failures++;
            Console.WriteLine($"[{name}] FAILED: decode error ({decodeError})");
        }
        else
        {
            Console.WriteLine($"[{name}] decode failed as expected ({decodeError})");
        }

        continue;
    }

    if (!expectTranslate)
    {
        failures++;
        Console.WriteLine(
            $"[{name}] FAILED: decoded successfully but is pinned as a decode failure — " +
            "if the new decode support is intentional, its semantics need verifying here first");
        continue;
    }

    // Buffer stores need a global-memory binding; the emitter resolves them by
    // instruction PC, so collect store PCs from the decoded program itself.
    var storePcs = new List<uint>();
    foreach (var instruction in program!.Instructions)
    {
        if (instruction.Opcode.StartsWith("BufferStore", StringComparison.Ordinal))
        {
            storePcs.Add(instruction.Pc);
        }
    }

    // The binding's scalar base (8 -> s[8:11]) must match the srsrc field of
    // the hand-assembled buffer_store words, and the 64-byte backing store
    // must cover every hand-assembled store offset.
    var globalBindings = storePcs.Count > 0
        ? new[]
        {
            new Gen5GlobalMemoryBinding(
                8u,
                0UL,
                storePcs,
                new byte[64],
                64,
                false),
        }
        : Array.Empty<Gen5GlobalMemoryBinding>();

    var state = new Gen5ShaderState(program, new uint[16], Metadata: null);
    var evaluation = new Gen5ShaderEvaluation(
        new uint[256],
        new uint[256],
        Array.Empty<Gen5ImageBinding>(),
        globalBindings);

    if (!name.StartsWith("lds-", StringComparison.Ordinal))
    {
        if (Gen5SpirvTranslator.TryCompileVertexShader(
                state,
                evaluation,
                out var vertexShader,
                out var vertexError))
        {
            var path = Path.Combine(outputDirectory, $"{name}.spv");
            File.WriteAllBytes(path, vertexShader.Spirv);
            Console.WriteLine($"[{name}] emit: success, {vertexShader.Spirv.Length} bytes -> {path}");
        }
        else
        {
            failures++;
            Console.WriteLine($"[{name}] emit: FAILED ({vertexError})");
        }
    }

    if (Gen5SpirvTranslator.TryCompileComputeShader(state, evaluation, 1, 1, 1, out var computeShader, out var computeError))
    {
        var path = Path.Combine(outputDirectory, $"{name}-cs.spv");
        File.WriteAllBytes(path, computeShader.Spirv);
        Console.WriteLine($"[{name}] compute emit: success, {computeShader.Spirv.Length} bytes -> {path}");
    }
    else
    {
        failures++;
        Console.WriteLine($"[{name}] compute emit: FAILED ({computeError})");
    }

    if (name.StartsWith("mrt", StringComparison.Ordinal))
    {
        Gen5PixelOutputBinding[] pixelOutputs = name switch
        {
            "mrt" =>
            [
                new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(3, 1, Gen5PixelOutputKind.Uint),
                new Gen5PixelOutputBinding(6, 2, Gen5PixelOutputKind.Sint),
            ],
            "mrt-float2" =>
            [
                new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(1, 1, Gen5PixelOutputKind.Float),
            ],
            "mrt8" =>
            [
                new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(1, 1, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(2, 2, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(3, 3, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(4, 4, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(5, 5, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(6, 6, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(7, 7, Gen5PixelOutputKind.Float),
            ],
            _ => [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float)],
        };

        if (Gen5SpirvTranslator.TryCompilePixelShader(state, evaluation, pixelOutputs, out var pixelShader, out var pixelError))
        {
            var path = Path.Combine(outputDirectory, $"{name}-ps.spv");
            File.WriteAllBytes(path, pixelShader.Spirv);
            Console.WriteLine($"[{name}] pixel emit: success, {pixelShader.Spirv.Length} bytes -> {path}");
        }
        else
        {
            failures++;
            Console.WriteLine($"[{name}] pixel emit: FAILED ({pixelError})");
        }

        if (name == "mrt")
        {
            Gen5PixelOutputBinding[] invalidOutputs =
            [
                new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float),
                new Gen5PixelOutputBinding(3, 7, Gen5PixelOutputKind.Float),
            ];
            if (Gen5SpirvTranslator.TryCompilePixelShader(state, evaluation, invalidOutputs, out _, out var invalidError))
            {
                failures++;
                Console.WriteLine("[mrt] FAILED: sparse host locations were accepted");
            }
            else
            {
                Console.WriteLine($"[mrt] sparse host locations rejected as expected ({invalidError})");
            }
        }
    }
}

Console.WriteLine(failures == 0
    ? "RESULT: all programs behaved as expected"
    : $"RESULT: {failures} unexpected outcome(s)");
Environment.ExitCode = failures == 0 ? 0 : 1;

internal sealed class FakeMemory : ICpuMemory
{
    private readonly List<(ulong Base, byte[] Data)> _regions = [];

    public void AddRegion(ulong baseAddress, uint[] words)
    {
        var bytes = new byte[words.Length * sizeof(uint)];
        for (var index = 0; index < words.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(
                bytes.AsSpan(index * sizeof(uint)),
                words[index]);
        }

        _regions.Add((baseAddress, bytes));
    }

    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        foreach (var (baseAddress, data) in _regions)
        {
            if (virtualAddress >= baseAddress &&
                virtualAddress + (ulong)destination.Length <= baseAddress + (ulong)data.Length)
            {
                data.AsSpan(
                    (int)(virtualAddress - baseAddress),
                    destination.Length).CopyTo(destination);
                return true;
            }
        }

        return false;
    }

    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;
}
