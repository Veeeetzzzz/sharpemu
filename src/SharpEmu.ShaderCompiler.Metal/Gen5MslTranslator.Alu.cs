// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private const string TauLiteral = "6.2831853071795862f";

        // ---- vector ALU ----

        private bool TryEmitVectorAlu(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Opcode is "VNop" or "VPipeflush" or "VClrexcp")
            {
                return true;
            }

            if (instruction.Control is Gen5SdwaControl sdwa &&
                (sdwa.Source0Select == 7 ||
                 sdwa.Source1Select == 7 ||
                 sdwa.DestinationSelect == 7 ||
                 sdwa.DestinationUnused == 3))
            {
                error = $"reserved SDWA selector/modifier in {instruction.Opcode}";
                return false;
            }

            if (instruction.Control is Gen5DppControl dppControl &&
                !IsSupportedDppControl(dppControl.Control))
            {
                error = $"unsupported DPP16 control 0x{dppControl.Control:X3}";
                return false;
            }

            if (instruction.Opcode.StartsWith("VCmp", StringComparison.Ordinal))
            {
                return TryEmitVectorCompare(instruction, out error);
            }

            switch (instruction.Opcode)
            {
                case "VReadfirstlaneB32":
                {
                    if (instruction.Destinations.Count == 0 ||
                        instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister ||
                        instruction.Sources.Count == 0)
                    {
                        error = "invalid read-first-lane operands";
                        return false;
                    }

                    // Under the single-lane graphics model the "first active
                    // lane" is always this lane; a real simd_shuffle would read
                    // another fragment's value. Compute broadcasts from the
                    // first guest-active lane (the ballot of EXEC), matching
                    // the SPIR-V translator — SPIR-V's own BroadcastFirst uses
                    // the first host-active invocation, which may be a lane the
                    // guest has masked off.
                    var value = RawSource(instruction, 0);
                    if (IsSingleLaneStage)
                    {
                        StoreScalar(instruction.Destinations[0].Value, Temp("uint", value));
                        return true;
                    }

                    if (IsWave64)
                    {
                        StoreScalar(
                            instruction.Destinations[0].Value,
                            EmitWave64ReadFirstLane(value));
                        return true;
                    }

                    var mask = Temp("uint", "sharpemu_ballot(exec)");
                    var firstLane = Temp("uint", $"{mask} == 0u ? 0u : (uint)ctz({mask})");
                    StoreScalar(
                        instruction.Destinations[0].Value,
                        Temp("uint", ShuffleLane(value, firstLane)));
                    return true;
                }
                case "VReadlaneB32":
                {
                    if (instruction.Destinations.Count == 0 ||
                        instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister)
                    {
                        error = "VReadlaneB32 expects scalar destination";
                        return false;
                    }

                    var value = RawSource(instruction, 0);
                    var lane = Temp("uint", $"({RawSource(instruction, 1)}) & 31u");
                    StoreScalar(
                        instruction.Destinations[0].Value,
                        Temp("uint", ShuffleLane(value, lane)));
                    return true;
                }
                case "VWritelaneB32":
                {
                    // vdst[lane(src1)] = src0; a writelane lands regardless of EXEC.
                    var destination = DestinationVector(instruction);
                    var source = RawSource(instruction, 0);
                    var lane = RawSource(instruction, 1);
                    StoreVector(
                        destination,
                        $"(sharpemu_lane == (({lane}) & 31u)) ? ({source}) : v[{destination}]",
                        guardWithExec: false);
                    return true;
                }
                case "VMovreldB32":
                case "VMovrelsB32":
                case "VMovrelsdB32":
                case "VMovrelsd2B32":
                    return TryEmitVectorRelativeMove(instruction, out error);
                case "VSwapB32":
                case "VSwaprelB32":
                    return TryEmitVectorSwap(instruction, out error);
                case "VLshlrevB64":
                case "VLshrrevB64":
                case "VAshrrevI64":
                    return TryEmitVector64Shift(instruction, out error);
                case "VQsadPkU16U8":
                case "VMqsadPkU16U8":
                case "VMqsadU32U8":
                    return TryEmitPackedSad(instruction, out error);
                case "VFrexpMantF64":
                    return TryEmitFrexpMantissaF64(instruction, out error);
                case "VCvtF64I32":
                    return TryEmitFloat64FromInt32(instruction, signed: true, out error);
                case "VCvtF64U32":
                    return TryEmitFloat64FromInt32(instruction, signed: false, out error);
                case "VCvtF64F32":
                    return TryEmitFloat64FromF32(instruction, out error);
                case "VTruncF64":
                    return TryEmitFloat64Round(instruction, Float64RoundMode.Trunc, out error);
                case "VCeilF64":
                    return TryEmitFloat64Round(instruction, Float64RoundMode.Ceil, out error);
                case "VRndneF64":
                    return TryEmitFloat64Round(instruction, Float64RoundMode.NearestEven, out error);
                case "VFloorF64":
                    return TryEmitFloat64Round(instruction, Float64RoundMode.Floor, out error);
                case "VFractF64":
                    return TryEmitFloat64Fract(instruction, out error);
                case "VCndmaskB32":
                {
                    // dst = mask-bit(lane) ? src1 : src0. Sources are raw (no
                    // float modifiers), matching the SPIR-V translator; the mask
                    // is VCC for VOP2 and an explicit SGPR operand for VOP3.
                    var mask = instruction.Sources.Count > 2
                        ? MaskBitExpression(instruction.Sources[2])
                        : "vcc";
                    StoreVector(
                        DestinationVector(instruction),
                        $"({mask}) ? ({RawSource(instruction, 1)}) : ({RawSource(instruction, 0)})");
                    return true;
                }
            }

            return TryEmitVectorValue(instruction, out error);
        }

        private bool TryEmitVectorRelativeMove(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Destinations.Count != 1 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.VectorRegister ||
                instruction.Sources.Count != 1)
            {
                error = $"invalid {instruction.Opcode} operands";
                return false;
            }

            var relativeSource = instruction.Opcode is
                "VMovrelsB32" or "VMovrelsdB32" or "VMovrelsd2B32";
            if (relativeSource &&
                instruction.Sources[0].Kind != Gen5OperandKind.VectorRegister)
            {
                error = $"{instruction.Opcode} expects a VGPR source base";
                return false;
            }

            const uint m0Register = 124;
            var splitOffsets = instruction.Opcode == "VMovrelsd2B32";
            var sourceOffset = splitOffsets
                ? $"(s[{m0Register}] & 0x3FFu)"
                : $"s[{m0Register}]";
            var destinationOffset = splitOffsets
                ? $"((s[{m0Register}] >> 16u) & 0x3FFu)"
                : $"s[{m0Register}]";
            var source = relativeSource
                ? LoadVectorRelative(instruction.Sources[0].Value, sourceOffset)
                : Temp("uint", RawSource(instruction, 0));
            var destination = instruction.Destinations[0].Value;
            if (instruction.Opcode is
                "VMovreldB32" or "VMovrelsdB32" or "VMovrelsd2B32")
            {
                StoreVectorRelative(destination, destinationOffset, source);
            }
            else
            {
                StoreVector(destination, source);
            }

            return true;
        }

        private bool TryEmitVector64Shift(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Destinations.Count == 0 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.VectorRegister ||
                instruction.Sources.Count < 2)
            {
                error = $"invalid {instruction.Opcode} operands";
                return false;
            }

            var destination = instruction.Destinations[0].Value;
            var shift = Temp("uint", $"({RawSource(instruction, 0)}) & 63u");
            var value = Temp("ulong", RawSource64(instruction, 1));
            var shifted = instruction.Opcode switch
            {
                "VLshlrevB64" => Temp("ulong", $"{value} << {shift}"),
                "VLshrrevB64" => Temp("ulong", $"{value} >> {shift}"),
                _ => Temp(
                    "ulong",
                    $"as_type<ulong>(as_type<long>({value}) >> {shift})"),
            };

            StoreVector(destination, $"(uint){shifted}");
            StoreVector(destination + 1, $"(uint)({shifted} >> 32u)");
            return true;
        }

        private bool TryEmitVectorSwap(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count != 2 ||
                instruction.Destinations.Count != 2 ||
                instruction.Sources.Any(item => item.Kind != Gen5OperandKind.VectorRegister) ||
                instruction.Destinations.Any(item => item.Kind != Gen5OperandKind.VectorRegister))
            {
                error = $"invalid {instruction.Opcode} operands";
                return false;
            }

            var sourceBase = instruction.Sources[0].Value;
            var destinationBase = instruction.Sources[1].Value;
            if (instruction.Opcode == "VSwapB32")
            {
                var sourceValue = Temp("uint", $"v[{sourceBase}]");
                var destinationValue = Temp("uint", $"v[{destinationBase}]");
                StoreVector(destinationBase, sourceValue);
                StoreVector(sourceBase, destinationValue);
                return true;
            }

            const uint m0Register = 124;
            var sourceOffset = $"(s[{m0Register}] & 0x3FFu)";
            var destinationOffset = $"((s[{m0Register}] >> 16u) & 0x3FFu)";
            var sourceValueRelative = LoadVectorRelative(sourceBase, sourceOffset);
            var destinationValueRelative = LoadVectorRelative(
                destinationBase,
                destinationOffset);
            StoreVectorRelative(destinationBase, destinationOffset, sourceValueRelative);
            StoreVectorRelative(sourceBase, sourceOffset, destinationValueRelative);
            return true;
        }

        private bool TryEmitVectorValue(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            var destination = DestinationVector(instruction);
            string? expression = instruction.Opcode switch
            {
                "VMovB32" => RawSource(instruction, 0),

                // ---- float arithmetic ----
                "VAddF32" => FloatResult(instruction, $"{F(instruction, 0)} + {F(instruction, 1)}"),
                "VAddF16" => Float16Result(
                    instruction,
                    destination,
                    $"{F16(instruction, 0)} + {F16(instruction, 1)}"),
                "VSubF32" => FloatResult(instruction, $"{F(instruction, 0)} - {F(instruction, 1)}"),
                "VSubrevF32" => FloatResult(instruction, $"{F(instruction, 1)} - {F(instruction, 0)}"),
                "VMulLegacyF32" => EmitLegacyFloatMultiply(instruction),
                "VMacLegacyF32" => EmitLegacyFloatMultiplyAccumulate(instruction, destination),
                "VMadLegacyF32" => EmitLegacyFloatMad(instruction),
                "VMullitF32" => EmitMullitF32(instruction),
                "VMulF32" => FloatResult(instruction, $"{F(instruction, 0)} * {F(instruction, 1)}"),
                "VMulF16" => Float16Result(
                    instruction,
                    destination,
                    $"{F16(instruction, 0)} * {F16(instruction, 1)}"),
                "VMinF32" => FloatResult(instruction, $"fmin({F(instruction, 0)}, {F(instruction, 1)})"),
                "VMaxF32" => FloatResult(instruction, $"fmax({F(instruction, 0)}, {F(instruction, 1)})"),
                "VMinF16" => Float16Result(
                    instruction,
                    destination,
                    $"fmin({F16(instruction, 0)}, {F16(instruction, 1)})"),
                "VMaxF16" => Float16Result(
                    instruction,
                    destination,
                    $"fmax({F16(instruction, 0)}, {F16(instruction, 1)})"),
                // The decoder normalizes mk/ak literal placement, so every MAD/FMA
                // form is fma(src0, src1, src2) exactly like the SPIR-V translator.
                "VFmaF32" or "VMadF32" or "VMadAkF32" or "VMadMkF32" or "VFmaAkF32" or "VFmaMkF32" =>
                    FloatResult(instruction, $"fma({F(instruction, 0)}, {F(instruction, 1)}, {F(instruction, 2)})"),
                "VFmacF32" or "VMacF32" =>
                    FloatResult(instruction, $"fma({F(instruction, 0)}, {F(instruction, 1)}, as_type<float>(v[{destination}]))"),
                "VFloorF32" => FloatResult(instruction, $"floor({F(instruction, 0)})"),
                "VCeilF32" => FloatResult(instruction, $"ceil({F(instruction, 0)})"),
                "VTruncF32" => FloatResult(instruction, $"trunc({F(instruction, 0)})"),
                "VRndneF32" => FloatResult(instruction, $"rint({F(instruction, 0)})"),
                "VFractF32" => FloatResult(instruction, $"fract({F(instruction, 0)})"),
                "VSqrtF32" => FloatResult(instruction, $"sqrt({F(instruction, 0)})"),
                "VRsqF32" => FloatResult(instruction, $"rsqrt({F(instruction, 0)})"),
                 "VRcpF32" or "VRcpIflagF32" => FloatResult(instruction, $"(1.0f / {F(instruction, 0)})"),
                 "VLogF32" => FloatResult(instruction, $"log2({F(instruction, 0)})"),
                 "VExpF32" => FloatResult(instruction, $"exp2({F(instruction, 0)})"),
                 "VFrexpExpI32F32" => EmitFrexpExponentF32(instruction),
                 "VFrexpMantF32" => FloatResult(
                     instruction,
                     $"as_type<float>({EmitFrexpMantissaF32(instruction)})"),
                 "VFrexpExpI32F64" => EmitFrexpExponentF64(instruction),
                // GCN sin/cos take revolutions; mirror the SPIR-V Tau prescale.
                "VSinF32" => FloatResult(instruction, $"sin({F(instruction, 0)} * {TauLiteral})"),
                "VCosF32" => FloatResult(instruction, $"cos({F(instruction, 0)} * {TauLiteral})"),
                "VLdexpF32" =>
                    FloatResult(instruction, $"ldexp({F(instruction, 0)}, as_type<int>({RawSource(instruction, 1)}))"),
                "VMin3F32" =>
                    FloatResult(instruction, $"fmin(fmin({F(instruction, 0)}, {F(instruction, 1)}), {F(instruction, 2)})"),
                "VMax3F32" =>
                    FloatResult(instruction, $"fmax(fmax({F(instruction, 0)}, {F(instruction, 1)}), {F(instruction, 2)})"),
                "VMed3F32" =>
                    FloatResult(instruction, $"fmax(fmin({F(instruction, 0)}, {F(instruction, 1)}), fmin(fmax({F(instruction, 0)}, {F(instruction, 1)}), {F(instruction, 2)}))"),

                // ---- conversions ----
                 "VCvtF32I32" => FloatResult(instruction, $"(float)as_type<int>({RawSource(instruction, 0)})"),
                 "VCvtF32U32" => FloatResult(instruction, $"(float)({RawSource(instruction, 0)})"),
                 "VCvtF32F64" => EmitFloat32FromF64(instruction),
                 "VCvtU32F32" => $"(uint)({F(instruction, 0)})",
                 "VCvtI32F32" => AsUInt($"(int)({F(instruction, 0)})"),
                 "VCvtI32F64" => EmitFloat64ToInt32(instruction, signed: true),
                 "VCvtU32F64" => EmitFloat64ToInt32(instruction, signed: false),
                // RPI rounds toward positive infinity; FLR toward negative.
                "VCvtRpiI32F32" => AsUInt($"(int)ceil({F(instruction, 0)})"),
                "VCvtFlrI32F32" => AsUInt($"(int)floor({F(instruction, 0)})"),
                "VCvtF32Ubyte0" => FloatResult(instruction, $"(float)(({RawSource(instruction, 0)}) & 0xFFu)"),
                "VCvtF32Ubyte1" => FloatResult(instruction, $"(float)((({RawSource(instruction, 0)}) >> 8) & 0xFFu)"),
                "VCvtF32Ubyte2" => FloatResult(instruction, $"(float)((({RawSource(instruction, 0)}) >> 16) & 0xFFu)"),
                "VCvtF32Ubyte3" => FloatResult(instruction, $"(float)((({RawSource(instruction, 0)}) >> 24) & 0xFFu)"),
                "VCvtF16F32" =>
                    $"((uint)as_type<ushort>(half({F(instruction, 0)})))",
                "VCvtF32F16" =>
                    AsUInt($"(float)as_type<half>((ushort)(({RawSource(instruction, 0)}) & 0xFFFFu))"),
                "VCvtOffF32I4" =>
                    AsUInt($"sharpemu_off_i4_table[({RawSource(instruction, 0)}) & 15u]"),
                "VCvtPkU8F32" =>
                    EmitCvtPkU8F32(instruction),
                "VCvtPkrtzF16F32" =>
                    EmitCvtPkrtzF16F32(instruction),
                "VCvtPknormI16F32" =>
                    $"pack_float_to_snorm2x16(float2({F(instruction, 0)}, {F(instruction, 1)}))",
                "VCvtPknormU16F32" =>
                    $"pack_float_to_unorm2x16(float2({F(instruction, 0)}, {F(instruction, 1)}))",
                "VCvtPkU16U32" or "VCvtPkI16I32" =>
                    $"((({RawSource(instruction, 0)}) & 0xFFFFu) | ((({RawSource(instruction, 1)}) & 0xFFFFu) << 16))",
                "VDot2cF32F16" => EmitDot2cF32F16(instruction, destination),
                "VSadU8" or "VSadHiU8" or "VSadU16" or "VSadU32" =>
                    EmitUnsignedSad(instruction),
                "VMsadU8" => EmitMaskedUnsignedSadU8(instruction),

                // ---- integer arithmetic ----
                "VAddU32" or "VAddI32" =>
                    $"(({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)}))",
                "VAddNcU32" or "VAddNcI32" =>
                    $"(({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)}))",
                "VSubU32" or "VSubI32" =>
                    $"(({RawSource(instruction, 0)}) - ({RawSource(instruction, 1)}))",
                "VSubNcU32" or "VSubNcI32" =>
                    $"(({RawSource(instruction, 0)}) - ({RawSource(instruction, 1)}))",
                "VSubrevU32" or "VSubrevI32" =>
                    $"(({RawSource(instruction, 1)}) - ({RawSource(instruction, 0)}))",
                "VSubrevNcU32" =>
                    $"(({RawSource(instruction, 1)}) - ({RawSource(instruction, 0)}))",
                // The SPIR-V translator treats the U24 multiply as a full 32-bit
                // multiply (only the Hi/Mad forms mask); mirror it exactly.
                "VMulLoU32" or "VMulLoI32" or "VMulU32U24" =>
                    $"(({RawSource(instruction, 0)}) * ({RawSource(instruction, 1)}))",
                "VMulI32I24" => EmitSigned24Product(instruction, high: false),
                "VMulHiI32I24" => EmitSigned24Product(instruction, high: true),
                "VMulHiU32" =>
                    $"mulhi({RawSource(instruction, 0)}, {RawSource(instruction, 1)})",
                "VMulHiU32U24" =>
                    $"mulhi(({RawSource(instruction, 0)}) & 0xFFFFFFu, ({RawSource(instruction, 1)}) & 0xFFFFFFu)",
                "VMulHiI32" =>
                    AsUInt($"mulhi(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)}))"),
                "VMadU32U24" =>
                    $"(((({RawSource(instruction, 0)}) & 0xFFFFFFu) * (({RawSource(instruction, 1)}) & 0xFFFFFFu)) + ({RawSource(instruction, 2)}))",
                "VMadI32I24" => EmitSigned24Mad(instruction),
                "VMadU32U16" or "VMadI32I16" =>
                    EmitMixedWidthMad16(instruction),
                "VAdd3U32" =>
                    $"(({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)}) + ({RawSource(instruction, 2)}))",
                "VAddLshlU32" =>
                    $"((({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)})) << (({RawSource(instruction, 2)}) & 31u))",
                "VLshlAddU32" =>
                    $"((({RawSource(instruction, 0)}) << (({RawSource(instruction, 1)}) & 31u)) + ({RawSource(instruction, 2)}))",
                "VMinU32" => $"min({RawSource(instruction, 0)}, {RawSource(instruction, 1)})",
                "VMaxU32" => $"max({RawSource(instruction, 0)}, {RawSource(instruction, 1)})",
                "VMinI32" =>
                    AsUInt($"min(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)}))"),
                "VMaxI32" =>
                    AsUInt($"max(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)}))"),
                "VMin3U32" =>
                    $"min(min({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), {RawSource(instruction, 2)})",
                "VMax3U32" =>
                    $"max(max({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), {RawSource(instruction, 2)})",
                "VMin3I32" =>
                    AsUInt($"min(min(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), as_type<int>({RawSource(instruction, 2)}))"),
                "VMax3I32" =>
                    AsUInt($"max(max(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), as_type<int>({RawSource(instruction, 2)}))"),
                "VMed3U32" =>
                    $"max(min({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), min(max({RawSource(instruction, 0)}, {RawSource(instruction, 1)}), {RawSource(instruction, 2)}))",
                "VMed3I32" =>
                    AsUInt($"max(min(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), min(max(as_type<int>({RawSource(instruction, 0)}), as_type<int>({RawSource(instruction, 1)})), as_type<int>({RawSource(instruction, 2)})))"),
                "VFmaF16" or
                "VMin3F16" or "VMin3I16" or "VMin3U16" or
                "VMax3F16" or "VMax3I16" or "VMax3U16" or
                "VMed3F16" or "VMed3I16" or "VMed3U16" =>
                    EmitVop3Half(instruction, destination),
                "VAddNcU16" or "VSubNcU16" or "VMulLoU16" or
                "VLshrrevB16" or "VAshrrevI16" or
                "VMaxU16" or "VMaxI16" or "VMinU16" or "VMinI16" or
                "VAddNcI16" or "VSubNcI16" or "VLshlrevB16" =>
                    EmitVop3Integer16(instruction, destination),
                "VMadU16" or "VMadI16" =>
                    EmitVop3Integer16(instruction, destination),
                "VDivFixupF16" => EmitDivFixupF16(instruction, destination),
                "VDivFmasF32" => EmitDivFmasF32(instruction),
                "VPackB32F16" or "VCvtPknormI16F16" or "VCvtPknormU16F16" =>
                    EmitVop3HalfPack(instruction),
                "VPkMadI16" or "VPkMulLoU16" or "VPkAddI16" or "VPkSubI16" or
                "VPkLshlrevB16" or "VPkLshrrevB16" or "VPkAshrrevI16" or
                "VPkMaxI16" or "VPkMinI16" or "VPkMadU16" or "VPkAddU16" or
                "VPkSubU16" or "VPkMaxU16" or "VPkMinU16" =>
                    EmitPackedInteger16(instruction),
                "VDot2I32I16" or "VDot2U32U16" or
                "VDot4I32I8" or "VDot4U32U8" or
                "VDot8I32I4" or "VDot8U32U4" =>
                    EmitPackedIntegerDot(instruction),
                "VDot2F32F16" => EmitPackedFloatDot(instruction),
                "VPkAddF16" or "VPkMulF16" or "VPkMinF16" or
                "VPkMaxF16" or "VPkFmaF16" =>
                    EmitPackedF16(instruction),
                "VFmaMixF32" or "VFmaMixloF16" or "VFmaMixhiF16" =>
                    EmitFmaMix(instruction, destination),
                "VAddF16" or "VSubF16" or "VSubrevF16" or "VMulF16" or
                "VFmacF16" or "VFmaMkF16" or "VFmaAkF16" or
                "VMaxF16" or "VMinF16" or "VLdexpF16" or
                "VRcpF16" or "VSqrtF16" or "VRsqF16" or
                "VLogF16" or "VExpF16" or "VFrexpMantF16" or
                "VFloorF16" or "VCeilF16" or "VTruncF16" or
                "VRndneF16" or "VFractF16" or "VSinF16" or "VCosF16" =>
                    EmitScalarF16(instruction, destination),
                "VCvtF16U16" or "VCvtF16I16" or
                "VCvtU16F16" or "VCvtI16F16" =>
                    EmitScalarF16Conversion(instruction, destination),
                "VFrexpExpI16F16" =>
                    MergeScalar16Result(
                        instruction,
                        destination,
                        EmitHalfFrexpExponentBits(EmitScalarF16OperandBits(instruction, 0))),
                "VCvtNormI16F16" =>
                    MergeScalar16Result(
                        instruction,
                        destination,
                        $"pack_float_to_snorm2x16(float2(float({EmitScalarF16Operand(instruction, 0)}), 0.0f))"),
                "VCvtNormU16F16" =>
                    MergeScalar16Result(
                        instruction,
                        destination,
                        $"pack_float_to_unorm2x16(float2(float({EmitScalarF16Operand(instruction, 0)}), 0.0f))"),
                "VSatPkU8I16" => EmitSatPkU8I16(instruction),
                "VPkFmacF16" => EmitPackedF16Accumulate(instruction, destination),

                // ---- bitwise ----
                "VAndB32" => $"(({RawSource(instruction, 0)}) & ({RawSource(instruction, 1)}))",
                "VOrB32" => $"(({RawSource(instruction, 0)}) | ({RawSource(instruction, 1)}))",
                "VXorB32" => $"(({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)}))",
                "VXnorB32" => $"~(({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)}))",
                "VNotB32" => $"~({RawSource(instruction, 0)})",
                "VAndOrB32" =>
                    $"((({RawSource(instruction, 0)}) & ({RawSource(instruction, 1)})) | ({RawSource(instruction, 2)}))",
                "VLerpU8" => EmitLerpU8(instruction),
                "VXadU32" =>
                    $"((({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)})) + ({RawSource(instruction, 2)}))",
                "VPermB32" =>
                    $"(sharpemu_perm_byte({RawSource(instruction, 0)}, {RawSource(instruction, 1)}, ({RawSource(instruction, 2)}) & 0xffu) | " +
                    $"(sharpemu_perm_byte({RawSource(instruction, 0)}, {RawSource(instruction, 1)}, (({RawSource(instruction, 2)}) >> 8u) & 0xffu) << 8u) | " +
                    $"(sharpemu_perm_byte({RawSource(instruction, 0)}, {RawSource(instruction, 1)}, (({RawSource(instruction, 2)}) >> 16u) & 0xffu) << 16u) | " +
                    $"(sharpemu_perm_byte({RawSource(instruction, 0)}, {RawSource(instruction, 1)}, (({RawSource(instruction, 2)}) >> 24u) & 0xffu) << 24u))",
                "VOr3U32" or "VOr3B32" =>
                    $"(({RawSource(instruction, 0)}) | ({RawSource(instruction, 1)}) | ({RawSource(instruction, 2)}))",
                "VLshlOrU32" or "VLshlOrB32" =>
                    $"((({RawSource(instruction, 0)}) << (({RawSource(instruction, 1)}) & 31u)) | ({RawSource(instruction, 2)}))",
                "VXor3B32" =>
                    $"(({RawSource(instruction, 0)}) ^ ({RawSource(instruction, 1)}) ^ ({RawSource(instruction, 2)}))",
                "VLshlB32" => $"(({RawSource(instruction, 0)}) << (({RawSource(instruction, 1)}) & 31u))",
                "VLshlrevB32" => $"(({RawSource(instruction, 1)}) << (({RawSource(instruction, 0)}) & 31u))",
                "VLshrB32" => $"(({RawSource(instruction, 0)}) >> (({RawSource(instruction, 1)}) & 31u))",
                "VLshrrevB32" => $"(({RawSource(instruction, 1)}) >> (({RawSource(instruction, 0)}) & 31u))",
                "VAshrI32" =>
                    AsUInt($"(as_type<int>({RawSource(instruction, 0)}) >> (({RawSource(instruction, 1)}) & 31u))"),
                "VAshrrevI32" =>
                    AsUInt($"(as_type<int>({RawSource(instruction, 1)}) >> (({RawSource(instruction, 0)}) & 31u))"),
                "VBfeU32" =>
                    $"extract_bits({RawSource(instruction, 0)}, ({RawSource(instruction, 1)}) & 31u, ({RawSource(instruction, 2)}) & 31u)",
                "VBfiB32" =>
                    $"((({RawSource(instruction, 0)}) & ({RawSource(instruction, 1)})) | (~({RawSource(instruction, 0)}) & ({RawSource(instruction, 2)})))",
                "VAlignbitB32" =>
                    $"uint(((ulong({RawSource(instruction, 0)}) << 32ul) | ulong({RawSource(instruction, 1)})) >> ulong(({RawSource(instruction, 2)}) & 31u))",
                "VAlignbyteB32" =>
                    $"((({RawSource(instruction, 2)}) & 31u) >= 8u ? 0u : uint(((ulong({RawSource(instruction, 0)}) << 32ul) | ulong({RawSource(instruction, 1)})) >> ulong((({RawSource(instruction, 2)}) & 31u) * 8u)))",
                "VBfmB32" =>
                    $"(((1u << (({RawSource(instruction, 0)}) & 31u)) - 1u) << (({RawSource(instruction, 1)}) & 31u))",
                "VBfrevB32" => $"reverse_bits({RawSource(instruction, 0)})",
                "VBcntU32B32" => $"(popcount({RawSource(instruction, 0)}) + ({RawSource(instruction, 1)}))",
                "VFfblB32" =>
                    $"(({RawSource(instruction, 0)}) == 0u ? 0xFFFFFFFFu : (uint)ctz({RawSource(instruction, 0)}))",
                "VFfbhU32" => EmitFfbh(instruction, signed: false),
                "VFfbhI32" => EmitFfbh(instruction, signed: true),

                // ---- wave / lane ----
                // mbcnt reads the mask dword the guest passes (no cross-lane
                // op), so only the per-lane thread-mask math differs by wave
                // size. Wave64 lanes 32..63 count the whole low half in mbcnt_lo
                // and their own partial in mbcnt_hi; a 1u << lane for lane>=32
                // would be undefined, so those are split out.
                "VMbcntLoU32B32" => IsWave64
                    ? $"((sharpemu_lane >= 32u ? popcount({RawSource(instruction, 0)}) : popcount(({RawSource(instruction, 0)}) & ((1u << sharpemu_lane) - 1u))) + ({RawSource(instruction, 1)}))"
                    : $"(popcount(({RawSource(instruction, 0)}) & ((1u << sharpemu_lane) - 1u)) + ({RawSource(instruction, 1)}))",
                "VMbcntHiU32B32" => IsWave64
                    ? $"((sharpemu_lane >= 32u ? popcount(({RawSource(instruction, 0)}) & ((1u << (sharpemu_lane - 32u)) - 1u)) : 0u) + ({RawSource(instruction, 1)}))"
                    // Wave32: the high mask half holds no lanes; pass the addend.
                    : RawSource(instruction, 1),
                "VPermlane16B32" => EmitPermlane16(instruction, exchangeRows: false),
                "VPermlanex16B32" => EmitPermlane16(instruction, exchangeRows: true),

                // ---- cube map helpers ----
                "VCubeidF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Id),
                "VCubescF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Sc),
                "VCubetcF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Tc),
                "VCubemaF32" => EmitCubeCoordinate(instruction, CubeCoordinate.Ma),

                _ => null,
            };

            if (expression is null)
            {
                switch (instruction.Opcode)
                {
                    case "VAddCoU32":
                    {
                        var left = Temp("uint", RawSource(instruction, 0));
                        var right = Temp("uint", RawSource(instruction, 1));
                        var sum = Temp("uint", $"{left} + {right}");
                        StoreCarryOut(instruction, $"{sum} < {left}");
                        expression = sum;
                        break;
                    }
                    case "VSubCoU32":
                    case "VSubrevCoU32":
                    {
                        var reverse = instruction.Opcode == "VSubrevCoU32";
                        var left = Temp("uint", RawSource(instruction, reverse ? 1 : 0));
                        var right = Temp("uint", RawSource(instruction, reverse ? 0 : 1));
                        StoreCarryOut(instruction, $"{left} < {right}");
                        expression = $"({left} - {right})";
                        break;
                    }
                    case "VAddcU32":
                    case "VAddCoCiU32":
                    {
                        var left = Temp("uint", RawSource(instruction, 0));
                        var right = Temp("uint", RawSource(instruction, 1));
                        var carryIn = instruction.Sources.Count > 2
                            ? MaskBitExpression(instruction.Sources[2])
                            : "vcc";
                        var partial = Temp("uint", $"{left} + {right}");
                        var sum = Temp("uint", $"{partial} + (({carryIn}) ? 1u : 0u)");
                        StoreCarryOut(instruction, $"({partial} < {left}) || ({sum} < {partial})");
                        expression = sum;
                        break;
                    }
                    case "VSubbU32":
                    case "VSubbrevU32":
                    {
                        var reverse = instruction.Opcode == "VSubbrevU32";
                        var left = Temp("uint", RawSource(instruction, reverse ? 1 : 0));
                        var right = Temp("uint", RawSource(instruction, reverse ? 0 : 1));
                        var borrowIn = instruction.Sources.Count > 2
                            ? MaskBitExpression(instruction.Sources[2])
                            : "vcc";
                        var borrow = Temp("uint", $"({borrowIn}) ? 1u : 0u");
                        var partial = Temp("uint", $"{left} - {right}");
                        StoreCarryOut(instruction, $"({left} < {right}) || ({partial} < {borrow})");
                        expression = $"({partial} - {borrow})";
                        break;
                    }
                    case "VMadU64U32":
                    {
                        // 64-bit product+addend into a VGPR pair, carry to SDST.
                        var product = Temp(
                            "ulong",
                            $"(ulong)({RawSource(instruction, 0)}) * (ulong)({RawSource(instruction, 1)})");
                        var addend = Temp("ulong", RawSource64(instruction, 2));
                        var wide = Temp("ulong", $"{product} + {addend}");
                        StoreCarryOut(instruction, $"{wide} < {addend}");
                        StoreVector(destination + 1, $"(uint)({wide} >> 32)");
                        expression = $"(uint){wide}";
                        break;
                    }
                    default:
                        error = $"unsupported vector opcode {instruction.Opcode}";
                        return false;
                }
            }

            var result = Temp("uint", expression);
            if (instruction.Control is Gen5DppControl dpp)
            {
                var writeEnabled = EmitDppWriteEnabled(dpp);
                result = Temp("uint", $"({writeEnabled}) ? {result} : v[{destination}]");
            }

            if (instruction.Control is Gen5SdwaControl { ScalarDestination: null } sdwaDestination)
            {
                result = ApplySdwaDestination(sdwaDestination, result, $"v[{destination}]");
            }

            StoreVector(destination, result);
            return true;
        }

        // V_SAD_* uses unsigned component differences and accumulates into src2.
        // A VOP3 integer clamp saturates an overflowing addition to UINT_MAX.
        private string EmitUnsignedSad(Gen5ShaderInstruction instruction)
        {
            var source0 = Temp("uint", RawSource(instruction, 0));
            var source1 = Temp("uint", RawSource(instruction, 1));
            var source2 = Temp("uint", RawSource(instruction, 2));
            var clamp = instruction.Control is Gen5Vop3Control { Clamp: true };

            string Component(string source, int shift, string mask) =>
                shift == 0
                    ? $"({source} & {mask})"
                    : $"(({source} >> {shift}) & {mask})";

            string AbsDiff(string left, string right) =>
                $"(max({left}, {right}) - min({left}, {right}))";

            string SumComponents(int componentBits, int componentCount)
            {
                var mask = componentBits == 8 ? "0xFFu" : "0xFFFFu";
                var terms = new string[componentCount];
                for (var component = 0; component < componentCount; component++)
                {
                    var left = Component(source0, component * componentBits, mask);
                    var right = Component(source1, component * componentBits, mask);
                    terms[component] = AbsDiff(left, right);
                }

                return Temp("uint", string.Join(" + ", terms));
            }

            var difference = instruction.Opcode switch
            {
                "VSadU8" => SumComponents(8, 4),
                "VSadHiU8" => Temp("uint", $"{SumComponents(8, 4)} << 16"),
                "VSadU16" => SumComponents(16, 2),
                "VSadU32" => Temp("uint", AbsDiff(source0, source1)),
                _ => "0u",
            };
            var sum = Temp("uint", $"{source2} + {difference}");
            return clamp ? $"({sum} < {source2} ? 0xFFFFFFFFu : {sum})" : sum;
        }

        // V_MSAD_U8 copies S2 and conditionally accumulates four unsigned byte
        // absolute differences.  A zero byte in S1 suppresses that component.
        private string EmitMaskedUnsignedSadU8(Gen5ShaderInstruction instruction)
        {
            var source0 = Temp("uint", RawSource(instruction, 0));
            var source1 = Temp("uint", RawSource(instruction, 1));
            var result = Temp("uint", RawSource(instruction, 2));

            for (var component = 0; component < 4; component++)
            {
                var shift = component * 8;
                var left = Temp(
                    "uint",
                    $"(({source0} >> {shift}) & 0xFFu)");
                var right = Temp(
                    "uint",
                    $"(({source1} >> {shift}) & 0xFFu)");
                var difference = Temp(
                    "uint",
                    $"(max({left}, {right}) - min({left}, {right}))");
                result = Temp(
                    "uint",
                    $"{result} + ({right} == 0u ? 0u : {difference})");
            }

            return result;
        }

        private bool TryEmitPackedSad(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Destinations.Count == 0 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.VectorRegister ||
                instruction.Sources.Count < 3)
            {
                error = $"invalid {instruction.Opcode} operands";
                return false;
            }

            var destination = instruction.Destinations[0].Value;
            var source0 = Temp("ulong", RawSource64(instruction, 0));
            var source1 = Temp("uint", RawSource(instruction, 1));
            var masked = instruction.Opcode is "VMqsadPkU16U8" or "VMqsadU32U8";

            string Source2Dword(int index) => instruction.Sources[2].Kind switch
            {
                Gen5OperandKind.VectorRegister => $"v[{instruction.Sources[2].Value + (uint)index}]",
                Gen5OperandKind.ScalarRegister => $"s[{instruction.Sources[2].Value + (uint)index}]",
                _ when index == 0 => RawSource(instruction, 2),
                _ => "0u",
            };

            string Source0Chunk(int index) => Temp(
                "uint",
                $"(uint)({source0} >> {(uint)(index * 8)}u)");

            string ByteSad(string source0Chunk)
            {
                var terms = new string[4];
                for (var component = 0; component < 4; component++)
                {
                    var shift = component * 8;
                    var left = Temp(
                        "uint",
                        $"(({source0Chunk} >> {shift}) & 0xFFu)");
                    var right = Temp(
                        "uint",
                        $"(({source1} >> {shift}) & 0xFFu)");
                    var difference =
                        $"(max({left}, {right}) - min({left}, {right}))";
                    terms[component] = masked
                        ? $"({right} == 0u ? 0u : {difference})"
                        : difference;
                }

                return Temp("uint", string.Join(" + ", terms));
            }

            if (instruction.Opcode is "VQsadPkU16U8" or "VMqsadPkU16U8")
            {
                var source2 = Temp("ulong", RawSource64(instruction, 2));
                var packed = new string[4];
                for (var component = 0; component < 4; component++)
                {
                    var accumulator = Temp(
                        "uint",
                        $"(uint)({source2} >> {(uint)(component * 16)}u) & 0xFFFFu");
                    packed[component] = Temp(
                        "uint",
                        $"(({accumulator} + {ByteSad(Source0Chunk(component))}) & 0xFFFFu)");
                }

                var low = Temp("uint", $"{packed[0]} | ({packed[1]} << 16)");
                var high = Temp("uint", $"{packed[2]} | ({packed[3]} << 16)");
                StoreVector(destination + 1, high);
                StoreVector(destination, low);
                return true;
            }

            for (var component = 0; component < 4; component++)
            {
                var result = Temp(
                    "uint",
                    $"{Source2Dword(component)} + {ByteSad(Source0Chunk(component))}");
                StoreVector(destination + (uint)component, result);
            }

            return true;
        }

        private string EmitCvtPkU8F32(Gen5ShaderInstruction instruction)
        {
            var converted = Temp("uint", $"(uint)({F(instruction, 0)})");
            var offset = Temp("uint", $"(({RawSource(instruction, 1)}) & 3u) << 3");
            var baseValue = Temp("uint", RawSource(instruction, 2));
            return $"(({baseValue} & ~(0xFFu << {offset})) | (({converted} & 0xFFu) << {offset}))";
        }

        private string EmitCvtPkrtzF16F32(Gen5ShaderInstruction instruction)
        {
            // Round-to-zero via mantissa truncation before the half conversion,
            // mirroring the SPIR-V translator's TruncateFloat32ForPack.
            var first = Temp(
                "float",
                $"as_type<float>(as_type<uint>({F(instruction, 0)}) & 0xFFFFE000u)");
            var second = Temp(
                "float",
                $"as_type<float>(as_type<uint>({F(instruction, 1)}) & 0xFFFFE000u)");
            return $"(((uint)as_type<ushort>(half({first}))) | (((uint)as_type<ushort>(half({second}))) << 16))";
        }

        private string EmitDot2cF32F16(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            // Materialize each packed source once. DPP remapping can involve
            // threadgroup synchronization, so re-emitting RawSource(src0) is
            // not merely redundant text generation.
            var source0 = Temp("uint", RawSource(instruction, 0));
            var source1 = Temp("uint", RawSource(instruction, 1));
            var source0Low = $"float(as_type<half>((ushort)({source0} & 0xFFFFu)))";
            var source0High = $"float(as_type<half>((ushort)({source0} >> 16)))";
            var source1Low = $"float(as_type<half>((ushort)({source1} & 0xFFFFu)))";
            var source1High = $"float(as_type<half>((ushort)({source1} >> 16)))";
            return FloatResult(
                instruction,
                $"as_type<float>(v[{destination}]) + " +
                $"{source0Low} * {source1Low} + {source0High} * {source1High}");
        }

        private string EmitScalarF16(Gen5ShaderInstruction instruction, uint destination)
        {
            var source0 = EmitScalarF16Operand(instruction, 0);
            var source1 = instruction.Sources.Count > 1
                ? EmitScalarF16Operand(instruction, 1)
                : string.Empty;
            string value = instruction.Opcode switch
            {
                "VAddF16" => $"({source0} + {source1})",
                "VSubF16" => $"({source0} - {source1})",
                "VSubrevF16" => $"({source1} - {source0})",
                "VMulF16" => $"({source0} * {source1})",
                "VFmacF16" =>
                    $"fma({source0}, {source1}, {EmitScalarF16Destination(instruction, destination)})",
                "VFmaMkF16" or "VFmaAkF16" =>
                    $"fma({source0}, {source1}, {EmitScalarF16Operand(instruction, 2)})",
                "VMaxF16" => $"fmax({source0}, {source1})",
                "VMinF16" => $"fmin({source0}, {source1})",
                "VLdexpF16" =>
                    $"half(ldexp(float({source0}), int(short({EmitScalarF16SourceBits(instruction, 1)}))))",
                "VRcpF16" => $"half(1.0f / float({source0}))",
                "VSqrtF16" => $"half(sqrt(float({source0})))",
                "VRsqF16" => $"half(1.0f / sqrt(float({source0})))",
                "VLogF16" => $"half(log2(float({source0})))",
                "VExpF16" => $"half(exp2(float({source0})))",
                "VFrexpMantF16" =>
                    $"as_type<half>((ushort)({EmitHalfFrexpMantissaBits(EmitScalarF16OperandBits(instruction, 0))}))",
                "VFloorF16" => $"half(floor(float({source0})))",
                "VCeilF16" => $"half(ceil(float({source0})))",
                "VTruncF16" => $"half(trunc(float({source0})))",
                "VRndneF16" => $"half(rint(float({source0})))",
                "VFractF16" => $"half(fract(float({source0})))",
                "VSinF16" => $"half(sin(float({source0}) * {TauLiteral}))",
                "VCosF16" => $"half(cos(float({source0}) * {TauLiteral}))",
                _ => source0,
            };

            return FinishScalarF16Result(instruction, destination, value);
        }

        private string FinishScalarF16Result(
            Gen5ShaderInstruction instruction,
            uint destination,
            string value)
        {
            var (outputModifier, clamp) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.OutputModifier, control.Clamp),
                Gen5SdwaControl control => (control.OutputModifier, control.Clamp),
                _ => (0u, false),
            };
            value = outputModifier switch
            {
                1 => $"(({value}) * half(2.0f))",
                2 => $"(({value}) * half(4.0f))",
                3 => $"(({value}) * half(0.5f))",
                _ => value,
            };
            if (clamp)
            {
                value = $"clamp({value}, half(0.0f), half(1.0f))";
            }

            var halfBits = Temp("uint", $"(uint)as_type<ushort>(half({value}))");
            return MergeScalar16Result(instruction, destination, halfBits);
        }

        private string MergeScalar16Result(
            Gen5ShaderInstruction instruction,
            uint destination,
            string halfBits)
        {
            halfBits = Temp("uint", $"(({halfBits}) & 0xFFFFu)");
            if (instruction.Control is Gen5Vop3Control vop3)
            {
                return (vop3.OperandSelect & 8) == 0
                    ? $"((v[{destination}] & 0xFFFF0000u) | {halfBits})"
                    : $"((v[{destination}] & 0x0000FFFFu) | ({halfBits} << 16))";
            }

            // Native 16-bit VOP1/VOP2 operations define the high half as zero.
            return halfBits;
        }

        private string EmitScalarF16Conversion(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            if (instruction.Opcode is "VCvtF16U16" or "VCvtF16I16")
            {
                var sourceBits = EmitScalarF16SourceBits(instruction, 0);
                var value = instruction.Opcode == "VCvtF16I16"
                    ? $"half((float)as_type<short>((ushort)({sourceBits})))"
                    : $"half((float)(ushort)({sourceBits}))";
                return FinishScalarF16Result(instruction, destination, value);
            }

            var source = EmitScalarF16Operand(instruction, 0);
            var signed = instruction.Opcode == "VCvtI16F16";
            var bounded = Temp(
                "float",
                $"(isnan(float({source})) ? 0.0f : " +
                $"clamp(float({source}), {(signed ? "-32768.0f" : "0.0f")}, " +
                $"{(signed ? "32767.0f" : "65535.0f")}))");
            var integer = signed
                ? $"((uint)(ushort)(short)trunc({bounded}))"
                : $"((uint)(ushort)trunc({bounded}))";
            return MergeScalar16Result(instruction, destination, integer);
        }

        private string EmitHalfFrexpMantissaBits(string halfBits)
        {
            var bits = Temp("uint", $"(({halfBits}) & 0xFFFFu)");
            var sign = Temp("uint", $"({bits} & 0x8000u)");
            var exponent = Temp("uint", $"({bits} & 0x7C00u)");
            var fraction = Temp("uint", $"({bits} & 0x03FFu)");
            var normalizedFraction = Temp(
                "uint",
                $"(({fraction} << (clz({fraction}) - 21u)) & 0x03FFu)");
            return
                $"(({exponent} == 0x7C00u || ({exponent} == 0u && {fraction} == 0u)) " +
                $"? {bits} : ({sign} | 0x3800u | ({exponent} == 0u ? {normalizedFraction} : {fraction})))";
        }

        private string EmitHalfFrexpExponentBits(string halfBits)
        {
            var bits = Temp("uint", $"(({halfBits}) & 0xFFFFu)");
            var exponent = Temp("uint", $"(({bits} >> 10u) & 0x1Fu)");
            var fraction = Temp("uint", $"({bits} & 0x03FFu)");
            var normalExponent = Temp("uint", $"({exponent} - 14u)");
            var subnormalExponent = Temp("uint", $"(8u - clz({fraction}))");
            return
                $"(({exponent} == 0x1Fu || ({exponent} == 0u && {fraction} == 0u)) " +
                $"? 0u : (({exponent} == 0u ? {subnormalExponent} : {normalExponent}) & 0xFFFFu))";
        }

        private string EmitFrexpExponentF32(Gen5ShaderInstruction instruction)
        {
            var bits = Temp("uint", $"as_type<uint>({F(instruction, 0)})");
            var exponent = Temp("uint", $"(({bits} >> 23u) & 0xFFu)");
            var fraction = Temp("uint", $"({bits} & 0x007FFFFFu)");
            var msb = Temp("uint", $"(31u - clz({fraction}))");
            var normalExponent = Temp("int", $"(int({exponent}) - 126)");
            var subnormalExponent = Temp("int", $"(int({msb}) - 148)");
            return Temp(
                "uint",
                $"as_type<uint>({exponent} == 0xFFu ? 0 : " +
                $"({exponent} == 0u ? ({fraction} == 0u ? 0 : {subnormalExponent}) : {normalExponent}))");
        }

        private string EmitFrexpMantissaF32(Gen5ShaderInstruction instruction)
        {
            var bits = Temp("uint", $"as_type<uint>({F(instruction, 0)})");
            var sign = Temp("uint", $"({bits} & 0x80000000u)");
            var exponent = Temp("uint", $"({bits} & 0x7F800000u)");
            var fraction = Temp("uint", $"({bits} & 0x007FFFFFu)");
            var msb = Temp("uint", $"(31u - clz({fraction}))");
            var shift = Temp("uint", $"(23u - {msb})");
            var normalizedFraction = Temp(
                "uint",
                $"(({fraction} << {shift}) & 0x007FFFFFu)");
            var normal = Temp("uint", $"{sign} | 0x3F000000u | {fraction}");
            var subnormal = Temp("uint", $"{sign} | 0x3F000000u | {normalizedFraction}");
            var finite = Temp(
                "uint",
                $"{exponent} == 0u ? ({fraction} == 0u ? {bits} : {subnormal}) : {normal}");
            return Temp(
                "uint",
                $"{exponent} == 0x7F800000u ? {bits} : {finite}");
        }

        private string EmitFrexpExponentF64(Gen5ShaderInstruction instruction)
        {
            var bits = Temp("ulong", Float64SourceBits(instruction, 0));
            var exponent = Temp("uint", $"(uint)(({bits} >> 52u) & 0x7FFul)");
            var fraction = Temp("ulong", $"({bits} & 0x000FFFFFFFFFFFFFul)");
            var low = Temp("uint", $"(uint){fraction}");
            var high = Temp("uint", $"(uint)({fraction} >> 32u)");
            var highMsb = Temp("int", $"(31 - int(clz({high})))");
            var lowMsb = Temp("int", $"(31 - int(clz({low})))");
            var msb = Temp("int", $"{high} != 0u ? ({highMsb} + 32) : {lowMsb}");
            var normalExponent = Temp("int", $"(int({exponent}) - 1022)");
            var subnormalExponent = Temp("int", $"({msb} - 1073)");
            return Temp(
                "uint",
                $"as_type<uint>({exponent} == 0x7FFu ? 0 : " +
                $"({exponent} == 0u ? ({fraction} == 0ul ? 0 : {subnormalExponent}) : {normalExponent}))");
        }

        private bool TryEmitFrexpMantissaF64(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            var destination = DestinationVector(instruction);
            var bits = Temp("ulong", Float64SourceBits(instruction, 0));
            var sign = Temp("ulong", $"({bits} & 0x8000000000000000ul)");
            var exponent = Temp("ulong", $"({bits} & 0x7FF0000000000000ul)");
            var fraction = Temp("ulong", $"({bits} & 0x000FFFFFFFFFFFFFul)");
            var low = Temp("uint", $"(uint){fraction}");
            var high = Temp("uint", $"(uint)({fraction} >> 32u)");
            var highMsb = Temp("int", $"(31 - int(clz({high})))");
            var lowMsb = Temp("int", $"(31 - int(clz({low})))");
            var msb = Temp("int", $"{high} != 0u ? ({highMsb} + 32) : {lowMsb}");
            var safeMsb = Temp("uint", $"({fraction} == 0ul ? 0u : (uint){msb})");
            var shift = Temp("uint", $"(52u - {safeMsb})");
            var normalizedFraction = Temp(
                "ulong",
                $"(({fraction} << {shift}) & 0x000FFFFFFFFFFFFFul)");
            var normal = Temp(
                "ulong",
                $"{sign} | 0x3FE0000000000000ul | {fraction}");
            var subnormal = Temp(
                "ulong",
                $"{sign} | 0x3FE0000000000000ul | {normalizedFraction}");
            var finite = Temp(
                "ulong",
                $"{exponent} == 0ul ? ({fraction} == 0ul ? {bits} : {subnormal}) : {normal}");
            var result = Temp(
                "ulong",
                $"{exponent} == 0x7FF0000000000000ul ? {bits} : {finite}");
            StoreVector(destination, $"(uint){result}");
            StoreVector(destination + 1, $"(uint)({result} >> 32u)");
            return true;
        }

        private string EmitFloat64ToInt32(
            Gen5ShaderInstruction instruction,
            bool signed)
        {
            // Match RDNA2's truncating/saturating conversion without requiring
            // Metal shader-float64 support.  All intermediate work stays in the
            // IEEE-754 bit representation.
            var bits = Temp("ulong", Float64SourceBits(instruction, 0));
            var sign = Temp("ulong", $"({bits} & 0x8000000000000000ul)");
            var exponent = Temp("uint", $"(uint)(({bits} >> 52u) & 0x7FFul)");
            var fraction = Temp("ulong", $"({bits} & 0x000FFFFFFFFFFFFFul)");
            var isNan = Temp(
                "bool",
                $"({exponent} == 0x7FFu && {fraction} != 0ul)");
            var isNegative = Temp("bool", $"{sign} != 0ul");
            var significand = Temp(
                "ulong",
                $"({fraction} | 0x0010000000000000ul)");
            var leftShift = Temp(
                "uint",
                $"({exponent} >= 1075u ? min({exponent} - 1075u, 63u) : 0u)");
            var rightShift = Temp(
                "uint",
                $"({exponent} < 1075u ? min(1075u - {exponent}, 63u) : 0u)");
            var leftMagnitude = Temp("ulong", $"{significand} << {leftShift}");
            var rightMagnitude = Temp("ulong", $"{significand} >> {rightShift}");
            var magnitude = Temp(
                "ulong",
                $"{exponent} >= 1075u ? {leftMagnitude} : {rightMagnitude}");
            var truncated = Temp("uint", $"(uint){magnitude}");
            var inRange = Temp("bool", $"{exponent} < 1054u");
            string finite;
            if (signed)
            {
                var signedMagnitude = Temp("uint", $"0u - {truncated}");
                var normal = Temp(
                    "uint",
                    $"{isNegative} ? {signedMagnitude} : {truncated}");
                var saturated = Temp(
                    "uint",
                    $"{isNegative} ? 0x80000000u : 0x7FFFFFFFu");
                finite = Temp(
                    "uint",
                    $"{inRange} ? {normal} : {saturated}");
            }
            else
            {
                var normal = Temp(
                    "uint",
                    $"{isNegative} ? 0u : {truncated}");
                var saturated = Temp(
                    "uint",
                    $"{isNegative} ? 0u : 0xFFFFFFFFu");
                finite = Temp(
                    "uint",
                    $"{inRange} ? {normal} : {saturated}");
            }

            return Temp("uint", $"{isNan} ? 0u : {finite}");
        }

        private bool TryEmitFloat64FromInt32(
            Gen5ShaderInstruction instruction,
            bool signed,
            out string error)
        {
            error = string.Empty;
            var destination = DestinationVector(instruction);
            var source = Temp("uint", RawSource(instruction, 0));
            var negative = signed
                ? Temp("bool", $"({source} & 0x80000000u) != 0u")
                : "false";
            var magnitude = signed
                ? Temp("uint", $"{negative} ? (0u - {source}) : {source}")
                : Temp("uint", source);
            var isZero = Temp("bool", $"{magnitude} == 0u");
            var msb = Temp("uint", $"(uint)(31 - int(clz({magnitude})))");
            var safeMsb = Temp("uint", $"{isZero} ? 0u : {msb}");
            var shift = Temp("uint", $"52u - {safeMsb}");
            var fraction = Temp(
                "ulong",
                $"(((ulong){magnitude} << {shift}) & 0x000FFFFFFFFFFFFFul)");
            var exponent = Temp("ulong", $"(ulong)(1023u + {safeMsb})");
            var signBits = signed
                ? Temp("ulong", $"{negative} ? 0x8000000000000000ul : 0ul")
                : "0ul";
            var result = Temp(
                "ulong",
                $"{isZero} ? 0ul : ({signBits} | ({exponent} << 52u) | {fraction})");
            StoreVector(destination, $"(uint){result}");
            StoreVector(destination + 1, $"(uint)({result} >> 32u)");
            return true;
        }

        private bool TryEmitFloat64FromF32(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            var destination = DestinationVector(instruction);
            var bits = Temp("uint", $"as_type<uint>({F(instruction, 0)})");
            var sign = Temp(
                "ulong",
                $"(ulong)({bits} >> 31u) << 63u");
            var exponent = Temp("uint", $"({bits} >> 23u) & 0xFFu");
            var fraction = Temp("uint", $"{bits} & 0x007FFFFFu");
            var isZero = Temp("bool", $"{fraction} == 0u");
            var msb = Temp("uint", $"(uint)(31 - int(clz({fraction})))");
            var safeMsb = Temp("uint", $"{isZero} ? 0u : {msb}");
            var subnormalFraction = Temp(
                "ulong",
                $"(((ulong){fraction} << (52u - {safeMsb})) & 0x000FFFFFFFFFFFFFul)");
            var normalFraction = Temp(
                "ulong",
                $"(ulong){fraction} << 29u");
            var normalExponent = Temp("ulong", $"(ulong)({exponent} + 896u)");
            var subnormalExponent = Temp("ulong", $"(ulong)({safeMsb} + 874u)");
            var normal = Temp(
                "ulong",
                $"{sign} | ({normalExponent} << 52u) | {normalFraction}");
            var subnormal = Temp(
                "ulong",
                $"{sign} | ({subnormalExponent} << 52u) | {subnormalFraction}");
            var finite = Temp(
                "ulong",
                $"{exponent} == 0u ? ({isZero} ? {sign} : {subnormal}) : {normal}");
            var special = Temp(
                "ulong",
                $"{sign} | 0x7FF0000000000000ul | ((ulong){fraction} << 29u)");
            var result = Temp(
                "ulong",
                $"{exponent} == 0xFFu ? {special} : {finite}");
            StoreVector(destination, $"(uint){result}");
            StoreVector(destination + 1, $"(uint)({result} >> 32u)");
            return true;
        }

        private string EmitFloat32FromF64(Gen5ShaderInstruction instruction)
        {
            var bits = Temp("ulong", Float64SourceBits(instruction, 0));
            var sign = Temp("uint", $"(uint)({bits} >> 32u) & 0x80000000u");
            var magnitudeBits = Temp("ulong", $"{bits} & 0x7FFFFFFFFFFFFFFFul");
            var exponent = Temp("uint", $"(uint)(({magnitudeBits} >> 52u) & 0x7FFul)");
            var fraction = Temp("ulong", $"{magnitudeBits} & 0x000FFFFFFFFFFFFFul");
            var significand = Temp("ulong", $"{fraction} | 0x0010000000000000ul");
            var normalRetained = Temp("uint", $"(uint)({significand} >> 29u)");
            var normalRemainder = Temp("ulong", $"{significand} & 0x1FFFFFFFul");
            var normalRound = Temp(
                "bool",
                $"{normalRemainder} > 0x10000000ul || " +
                $"({normalRemainder} == 0x10000000ul && ({normalRetained} & 1u) != 0u)");
            var normalRounded = Temp("uint", $"{normalRetained} + ({normalRound} ? 1u : 0u)");
            var normalCarry = Temp("bool", $"{normalRounded} >= 0x01000000u");
            var normalExponent = Temp(
                "uint",
                $"({exponent} - 896u) + ({normalCarry} ? 1u : 0u)");
            var normalFraction = Temp(
                "uint",
                $"(({normalCarry} ? ({normalRounded} >> 1u) : {normalRounded}) & 0x007FFFFFu)");
            var normalBits = Temp(
                "uint",
                $"{exponent} >= 1151u ? ({sign} | 0x7F800000u) : " +
                $"({sign} | (({normalExponent} & 0xFFu) << 23u) | {normalFraction})");

            var subnormalShift = Temp("uint", $"min(926u - {exponent}, 63u)");
            var subnormalRetained = Temp(
                "uint",
                $"(uint)({significand} >> {subnormalShift})");
            var subnormalMask = Temp("ulong", $"(1ul << {subnormalShift}) - 1ul");
            var subnormalRemainder = Temp("ulong", $"{significand} & {subnormalMask}");
            var subnormalHalf = Temp("ulong", $"1ul << ({subnormalShift} - 1u)");
            var subnormalRound = Temp(
                "bool",
                $"{subnormalRemainder} > {subnormalHalf} || " +
                $"({subnormalRemainder} == {subnormalHalf} && ({subnormalRetained} & 1u) != 0u)");
            var subnormalRounded = Temp(
                "uint",
                $"{subnormalRetained} + ({subnormalRound} ? 1u : 0u)");
            var subnormalBits = Temp(
                "uint",
                $"{sign} | ({subnormalRounded} >= 0x00800000u ? 0x00800000u : {subnormalRounded})");
            var finite = Temp(
                "uint",
                $"{exponent} == 0u ? {sign} : ({exponent} < 897u ? {subnormalBits} : {normalBits})");
            var specialPayload = Temp("uint", $"(uint)({fraction} >> 29u) & 0x007FFFFFu");
            var specialFraction = Temp(
                "uint",
                $"{fraction} == 0ul ? 0u : ({specialPayload} == 0u ? 1u : {specialPayload})");
            var specialBits = Temp(
                "uint",
                $"{sign} | 0x7F800000u | {specialFraction}");
            return Temp(
                "uint",
                $"{exponent} == 0x7FFu ? {specialBits} : {finite}");
        }

        private enum Float64RoundMode
        {
            Trunc,
            Ceil,
            NearestEven,
            Floor,
        }

        private bool TryEmitFloat64Round(
            Gen5ShaderInstruction instruction,
            Float64RoundMode mode,
            out string error)
        {
            error = string.Empty;
            var destination = DestinationVector(instruction);
            var bits = Temp("ulong", Float64SourceBits(instruction, 0));
            var sign = Temp("ulong", $"({bits} & 0x8000000000000000ul)");
            var magnitudeBits = Temp("ulong", $"({bits} & 0x7FFFFFFFFFFFFFFFul)");
            var exponent = Temp("uint", $"(uint)(({magnitudeBits} >> 52u) & 0x7FFul)");
            var fraction = Temp("ulong", $"({magnitudeBits} & 0x000FFFFFFFFFFFFFul)");
            var significand = Temp("ulong", $"{fraction} | 0x0010000000000000ul");
            var isSpecial = Temp("bool", $"{exponent} == 0x7FFu");
            var isSubnormal = Temp("bool", $"{exponent} < 1023u");
            var hasNormalFraction = Temp("bool", $"{exponent} < 1075u");
            var normalShift = Temp("uint", $"1075u - {exponent}");
            var normalMask = Temp(
                "ulong",
                $"(((1ul << {normalShift}) - 1ul) & 0x000FFFFFFFFFFFFFul)");
            var truncMask = Temp(
                "ulong",
                $"{isSubnormal} ? 0x000FFFFFFFFFFFFFul : ({hasNormalFraction} ? {normalMask} : 0ul)");
            var truncatedMagnitude = Temp(
                "ulong",
                $"{magnitudeBits} & ~{truncMask}");
            var hasFraction = Temp(
                "bool",
                $"({magnitudeBits} & {truncMask}) != 0ul");
            string increment;
            switch (mode)
            {
                case Float64RoundMode.Trunc:
                    increment = "false";
                    break;
                case Float64RoundMode.Ceil:
                    increment = Temp("bool", $"({sign} == 0ul) && {hasFraction}");
                    break;
                case Float64RoundMode.Floor:
                    increment = Temp("bool", $"({sign} != 0ul) && {hasFraction}");
                    break;
                default:
                {
                    var isAtLeastHalf = Temp("bool", $"{exponent} >= 1022u");
                    var isRoundable = Temp("bool", $"{exponent} < 1075u");
                    var halfShift = Temp("uint", $"1075u - {exponent}");
                    var halfMask = Temp("ulong", $"(1ul << {halfShift}) - 1ul");
                    var remainder = Temp("ulong", $"{significand} & {halfMask}");
                    var half = Temp("ulong", $"1ul << ({halfShift} - 1u)");
                    var greaterHalf = Temp("bool", $"{remainder} > {half}");
                    var equalHalf = Temp("bool", $"{remainder} == {half}");
                    var odd = Temp(
                        "bool",
                        $"({exponent} < 1075u) && ((({significand} >> {halfShift}) & 1ul) != 0ul)");
                    increment = Temp(
                        "bool",
                        $"{isRoundable} && {isAtLeastHalf} && ({greaterHalf} || ({equalHalf} && {odd}))");
                    break;
                }
            }

            var unit = Temp(
                "ulong",
                $"{isSubnormal} ? 0x3FF0000000000000ul : ({hasNormalFraction} ? (1ul << {normalShift}) : 0ul)");
            var roundedMagnitude = Temp(
                "ulong",
                $"{increment} ? ({truncatedMagnitude} + {unit}) : {truncatedMagnitude}");
            var result = Temp(
                "ulong",
                $"{isSpecial} ? {bits} : ({sign} | {roundedMagnitude})");
            StoreVector(destination, $"(uint){result}");
            StoreVector(destination + 1, $"(uint)({result} >> 32u)");
            return true;
        }

        private bool TryEmitFloat64Fract(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            var destination = DestinationVector(instruction);
            var bits = Temp("ulong", Float64SourceBits(instruction, 0));
            var sign = Temp("ulong", $"{bits} & 0x8000000000000000ul");
            var magnitude = Temp("ulong", $"{bits} & 0x7FFFFFFFFFFFFFFFul");
            var exponent = Temp("uint", $"(uint)(({magnitude} >> 52u) & 0x7FFul)");
            var fraction = Temp("ulong", $"{magnitude} & 0x000FFFFFFFFFFFFFul");
            var significand = Temp("ulong", $"{fraction} | 0x0010000000000000ul");
            var belowOne = Temp("bool", $"{exponent} < 1023u");
            var belowInteger = Temp("bool", $"{exponent} < 1075u");
            var shift = Temp("uint", $"min(1075u - {exponent}, 63u)");
            var remainderMask = Temp("ulong", $"(1ul << {shift}) - 1ul");
            var remainder = Temp("ulong", $"{significand} & {remainderMask}");
            var remainderLow = Temp("uint", $"(uint){remainder}");
            var remainderHigh = Temp("uint", $"(uint)({remainder} >> 32u)");
            var remainderHighMsb = Temp("int", $"31 - int(clz({remainderHigh}))");
            var remainderLowMsb = Temp("int", $"31 - int(clz({remainderLow}))");
            var remainderMsb = Temp(
                "uint",
                $"{remainder} == 0ul ? 0u : ({remainderHigh} != 0u ? (uint)({remainderHighMsb} + 32) : (uint){remainderLowMsb})");
            var remainderFraction = Temp(
                "ulong",
                $"(({remainder} << (52u - {remainderMsb})) & 0x000FFFFFFFFFFFFFul)");
            var remainderExponent = Temp(
                "ulong",
                $"(ulong)({exponent} - 52u + {remainderMsb})");
            var normalizedRemainder = Temp(
                "ulong",
                $"({remainderExponent} << 52u) | {remainderFraction}");
            var positiveFraction = Temp(
                "ulong",
                $"{belowOne} ? {magnitude} : ({belowInteger} ? {normalizedRemainder} : 0ul)");

            var yExponent = Temp("uint", $"(uint)({positiveFraction} >> 52u) & 0x7FFu");
            var yFraction = Temp("ulong", $"{positiveFraction} & 0x000FFFFFFFFFFFFFul");
            var ySignificand = Temp("ulong", $"{yFraction} | 0x0010000000000000ul");
            var ySmall = Temp("bool", $"{yExponent} < 970u");
            var yShift = Temp("uint", $"min(1022u - {yExponent}, 63u)");
            var yMask = Temp("ulong", $"(1ul << {yShift}) - 1ul");
            var yRetained = Temp("ulong", $"{ySignificand} >> {yShift}");
            var yRemainder = Temp("ulong", $"{ySignificand} & {yMask}");
            var yHalf = Temp(
                "ulong",
                $"{yShift} == 0u ? 0ul : (1ul << ({yShift} - 1u))");
            var yRound = Temp(
                "bool",
                $"{yRemainder} > {yHalf} || ({yRemainder} == {yHalf} && ({yRetained} & 1ul) != 0ul)");
            var yUnits = Temp(
                "ulong",
                $"{ySmall} ? 0ul : ({yRetained} + ({yRound} ? 1ul : 0ul))");
            var halfTie = Temp(
                "bool",
                $"{yExponent} == 970u && {yFraction} == 0ul");
            var difference = Temp(
                "ulong",
                $"{halfTie} ? 0x0020000000000000ul : (0x0020000000000000ul - {yUnits})");
            var differenceLow = Temp("uint", $"(uint){difference}");
            var differenceHigh = Temp("uint", $"(uint)({difference} >> 32u)");
            var differenceHighMsb = Temp("int", $"31 - int(clz({differenceHigh}))");
            var differenceLowMsb = Temp("int", $"31 - int(clz({differenceLow}))");
            var differenceMsb = Temp(
                "uint",
                $"{differenceHigh} != 0u ? (uint)({differenceHighMsb} + 32) : (uint){differenceLowMsb}");
            var differenceSafeMsb = Temp(
                "uint",
                $"{difference} == 0x0020000000000000ul ? 52u : {differenceMsb}");
            var oneMinus = Temp(
                "ulong",
                $"{difference} == 0x0020000000000000ul ? 0x3FF0000000000000ul : " +
                $"(((ulong)(970u + {differenceSafeMsb}) << 52u) | " +
                $"(({difference} << (52u - {differenceSafeMsb})) & 0x000FFFFFFFFFFFFFul))");
            var negativeFraction = Temp(
                "ulong",
                $"{positiveFraction} != 0ul ? {oneMinus} : 0ul");
            var finite = Temp(
                "ulong",
                $"{sign} != 0ul ? {negativeFraction} : {positiveFraction}");
            var special = Temp(
                "ulong",
                $"{fraction} == 0ul ? ({sign} | 0x7FF8000000000000ul) : {bits}");
            var result = Temp(
                "ulong",
                $"{exponent} == 0x7FFu ? {special} : {finite}");
            StoreVector(destination, $"(uint){result}");
            StoreVector(destination + 1, $"(uint)({result} >> 32u)");
            return true;
        }

        private string EmitSatPkU8I16(Gen5ShaderInstruction instruction)
        {
            var source = Temp(
                "uint",
                RawSource(instruction, 0, applySdwaIntegerModifiers: false));
            var low = Temp(
                "uint",
                $"(uint)clamp(int(as_type<short>((ushort)({source} & 0xFFFFu))), 0, 255)");
            var high = Temp(
                "uint",
                $"(uint)clamp(int(as_type<short>((ushort)({source} >> 16u))), 0, 255)");
            return $"(({low} & 0xFFu) | (({high} & 0xFFu) << 8u))";
        }

        private string EmitFfbh(Gen5ShaderInstruction instruction, bool signed)
        {
            var source = Temp("uint", RawSource(instruction, 0));
            var search = signed
                ? Temp("uint", $"((({source} & 0x80000000u) != 0u) ? ~{source} : {source})")
                : source;
            return $"({search} == 0u ? 0xFFFFFFFFu : (uint)clz({search}))";
        }

        private string EmitScalarF16SourceBits(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            string raw;
            if (instruction.Sources[sourceIndex] is
                { Kind: Gen5OperandKind.EncodedConstant, Value: >= 240 and <= 248 } constant &&
                Gen5InlineConstants.TryDecode(constant.Value, out var floatBits))
            {
                raw = $"0x{BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(floatBits)):X4}u";
            }
            else
            {
                raw = RawSource(instruction, sourceIndex, applySdwaIntegerModifiers: false);
            }

            if (instruction.Control is Gen5Vop3Control control &&
                (control.OperandSelect & (1u << sourceIndex)) != 0)
            {
                raw = $"(({raw}) >> 16)";
            }

            return $"(({raw}) & 0xFFFFu)";
        }

        private string EmitScalarF16Operand(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            return Temp(
                "half",
                $"as_type<half>((ushort)({EmitScalarF16OperandBits(instruction, sourceIndex)}))");
        }

        private string EmitScalarF16OperandBits(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            string value = EmitScalarF16SourceBits(instruction, sourceIndex);
            var (absoluteMask, negateMask) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.AbsoluteMask, control.NegateMask),
                Gen5SdwaControl control => (control.AbsoluteMask, control.NegateMask),
                Gen5DppControl control => (control.AbsoluteMask, control.NegateMask),
                _ => (0u, 0u),
            };
            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                value = $"(({value}) & 0x7FFFu)";
            }

            if ((negateMask & (1u << sourceIndex)) != 0)
            {
                value = $"(({value}) ^ 0x8000u)";
            }

            return Temp("uint", value);
        }

        private string EmitScalarF16Destination(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            var bits = instruction.Control is Gen5Vop3Control { OperandSelect: var operandSelect } &&
                       (operandSelect & 8) != 0
                ? $"((v[{destination}] >> 16) & 0xFFFFu)"
                : $"(v[{destination}] & 0xFFFFu)";
            return $"as_type<half>((ushort)({bits}))";
        }

        private string EmitPackedF16Accumulate(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            var source0 = Temp("uint", RawSource(instruction, 0));
            var source1 = Temp("uint", RawSource(instruction, 1));
            string Lane(string source, bool high) =>
                $"as_type<half>((ushort)(({source} >> {(high ? 16 : 0)}) & 0xFFFFu))";
            var low = Temp(
                "half",
                $"fma({Lane(source0, false)}, {Lane(source1, false)}, " +
                $"as_type<half>((ushort)(v[{destination}] & 0xFFFFu)))");
            var high = Temp(
                "half",
                $"fma({Lane(source0, true)}, {Lane(source1, true)}, " +
                $"as_type<half>((ushort)(v[{destination}] >> 16)))");
            return $"((uint)as_type<ushort>({low}) | ((uint)as_type<ushort>({high}) << 16))";
        }

        /// <summary>
        /// Emits the scalar f16/i16/u16 VOP3 family. OPSEL[0:2] chooses a source
        /// half and OPSEL[3] chooses the destination half, preserving its sibling.
        /// </summary>
        private string EmitVop3Half(Gen5ShaderInstruction instruction, uint destination)
        {
            if (instruction.Control is not Gen5Vop3Control control)
            {
                throw new NotSupportedException($"missing VOP3 control for {instruction.Opcode}");
            }

            string halfBits;
            if (instruction.Opcode.EndsWith("F16", StringComparison.Ordinal))
            {
                var source0 = EmitVop3F16Operand(instruction, control, 0);
                var source1 = EmitVop3F16Operand(instruction, control, 1);
                var source2 = EmitVop3F16Operand(instruction, control, 2);
                string value = instruction.Opcode switch
                {
                    "VFmaF16" => $"fma({source0}, {source1}, {source2})",
                    "VMin3F16" => $"fmin(fmin({source0}, {source1}), {source2})",
                    "VMax3F16" => $"fmax(fmax({source0}, {source1}), {source2})",
                    "VMed3F16" => EmitVop3F16Median(source0, source1, source2),
                    _ => source0,
                };
                value = control.OutputModifier switch
                {
                    1 => $"(({value}) * half(2.0f))",
                    2 => $"(({value}) * half(4.0f))",
                    3 => $"(({value}) * half(0.5f))",
                    _ => value,
                };
                if (control.Clamp)
                {
                    value = $"clamp({value}, half(0.0f), half(1.0f))";
                }

                halfBits = $"(uint)as_type<ushort>(half({value}))";
            }
            else
            {
                var signed = instruction.Opcode.EndsWith("I16", StringComparison.Ordinal);
                var source0 = EmitVop3Integer16Operand(instruction, control, 0, signed);
                var source1 = EmitVop3Integer16Operand(instruction, control, 1, signed);
                var source2 = EmitVop3Integer16Operand(instruction, control, 2, signed);
                var min = instruction.Opcode.StartsWith("VMin3", StringComparison.Ordinal);
                var max = instruction.Opcode.StartsWith("VMax3", StringComparison.Ordinal);
                string value;
                if (min)
                {
                    value = $"min(min({source0}, {source1}), {source2})";
                }
                else if (max)
                {
                    value = $"max(max({source0}, {source1}), {source2})";
                }
                else
                {
                    var max01 = Temp(signed ? "short" : "ushort", $"max({source0}, {source1})");
                    var max3 = Temp(signed ? "short" : "ushort", $"max({max01}, {source2})");
                    value = $"({max3} == {source0} ? max({source1}, {source2}) : " +
                        $"({max3} == {source1} ? max({source0}, {source2}) : {max01}))";
                }

                halfBits = signed
                    ? $"(uint)as_type<ushort>(short({value}))"
                    : $"(uint)ushort({value})";
            }

            var packed = Temp("uint", $"({halfBits}) & 0xFFFFu");
            return (control.OperandSelect & 8) == 0
                ? $"((v[{destination}] & 0xFFFF0000u) | {packed})"
                : $"((v[{destination}] & 0x0000FFFFu) | ({packed} << 16))";
        }

        private string EmitVop3HalfBits(
            Gen5ShaderInstruction instruction,
            Gen5Vop3Control control,
            int sourceIndex)
        {
            string raw;
            if (instruction.Sources[sourceIndex] is
                { Kind: Gen5OperandKind.EncodedConstant, Value: >= 240 and <= 248 } constant &&
                Gen5InlineConstants.TryDecode(constant.Value, out var floatBits))
            {
                var half = (Half)BitConverter.UInt32BitsToSingle(floatBits);
                raw = $"0x{BitConverter.HalfToUInt16Bits(half):X4}u";
            }
            else
            {
                raw = RawSource(instruction, sourceIndex);
            }

            return (control.OperandSelect & (1u << sourceIndex)) == 0
                ? $"(({raw}) & 0xFFFFu)"
                : $"((({raw}) >> 16) & 0xFFFFu)";
        }

        private string EmitVop3F16Operand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3Control control,
            int sourceIndex)
        {
            string value = $"as_type<half>((ushort)({EmitVop3HalfBits(instruction, control, sourceIndex)}))";
            if ((control.AbsoluteMask & (1u << sourceIndex)) != 0)
            {
                value = $"abs({value})";
            }

            if ((control.NegateMask & (1u << sourceIndex)) != 0)
            {
                value = $"(-{value})";
            }

            return Temp("half", value);
        }

        private string EmitVop3F16Median(string source0, string source1, string source2)
        {
            var min3 = Temp("half", $"fmin(fmin({source0}, {source1}), {source2})");
            var max01 = Temp("half", $"fmax({source0}, {source1})");
            var max3 = Temp("half", $"fmax({max01}, {source2})");
            var median = $"({max3} == {source0} ? fmax({source1}, {source2}) : " +
                $"({max3} == {source1} ? fmax({source0}, {source2}) : {max01}))";
            return $"(isnan({source0}) || isnan({source1}) || isnan({source2}) ? {min3} : {median})";
        }

        private string EmitVop3Integer16Operand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3Control control,
            int sourceIndex,
            bool signed)
        {
            var bits = EmitVop3HalfBits(instruction, control, sourceIndex);
            return Temp(signed ? "short" : "ushort", $"({(signed ? "short" : "ushort")})({bits})");
        }

        private string EmitVop3Integer16(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            if (instruction.Control is not Gen5Vop3Control control)
            {
                throw new NotSupportedException($"missing VOP3 control for {instruction.Opcode}");
            }

            var leftBits = EmitVop3HalfBits(instruction, control, 0);
            var rightBits = EmitVop3HalfBits(instruction, control, 1);
            var leftUnsigned = Temp("uint", leftBits);
            var rightUnsigned = Temp("uint", rightBits);
            var leftSigned = Temp("int", $"(int)(short)({leftBits})");
            var rightSigned = Temp("int", $"(int)(short)({rightBits})");
            string value = instruction.Opcode switch
            {
                "VAddNcU16" => control.Clamp
                    ? $"min({leftUnsigned} + {rightUnsigned}, 65535u)"
                    : $"{leftUnsigned} + {rightUnsigned}",
                "VSubNcU16" => control.Clamp
                    ? $"({leftUnsigned} < {rightUnsigned} ? 0u : {leftUnsigned} - {rightUnsigned})"
                    : $"{leftUnsigned} - {rightUnsigned}",
                "VMulLoU16" => control.Clamp
                    ? $"min({leftUnsigned} * {rightUnsigned}, 65535u)"
                    : $"{leftUnsigned} * {rightUnsigned}",
                "VLshrrevB16" => $"{rightUnsigned} >> ({leftUnsigned} & 15u)",
                "VLshlrevB16" => $"{rightUnsigned} << ({leftUnsigned} & 15u)",
                "VAshrrevI16" => $"(uint)({rightSigned} >> ({leftUnsigned} & 15u))",
                "VMaxU16" => $"max({leftUnsigned}, {rightUnsigned})",
                "VMinU16" => $"min({leftUnsigned}, {rightUnsigned})",
                "VMaxI16" => $"(uint)max({leftSigned}, {rightSigned})",
                "VMinI16" => $"(uint)min({leftSigned}, {rightSigned})",
                "VAddNcI16" => control.Clamp
                    ? $"(uint)clamp({leftSigned} + {rightSigned}, -32768, 32767)"
                    : $"(uint)({leftSigned} + {rightSigned})",
                "VSubNcI16" => control.Clamp
                    ? $"(uint)clamp({leftSigned} - {rightSigned}, -32768, 32767)"
                    : $"(uint)({leftSigned} - {rightSigned})",
                "VMadU16" => control.Clamp
                    ? $"min({leftUnsigned} * {rightUnsigned} + " +
                      $"(uint)({EmitVop3HalfBits(instruction, control, 2)}), 65535u)"
                    : $"{leftUnsigned} * {rightUnsigned} + " +
                      $"(uint)({EmitVop3HalfBits(instruction, control, 2)})",
                "VMadI16" => control.Clamp
                    ? $"(uint)clamp({leftSigned} * {rightSigned} + " +
                      $"(int)(short)({EmitVop3HalfBits(instruction, control, 2)}), -32768, 32767)"
                    : $"(uint)({leftSigned} * {rightSigned} + " +
                      $"(int)(short)({EmitVop3HalfBits(instruction, control, 2)}))",
                _ => throw new NotSupportedException($"unsupported VOP3 i16 operation {instruction.Opcode}"),
            };

            var packed = Temp("uint", $"({value}) & 0xFFFFu");
            return (control.OperandSelect & 8) == 0
                ? $"((v[{destination}] & 0xFFFF0000u) | {packed})"
                : $"((v[{destination}] & 0x0000FFFFu) | ({packed} << 16))";
        }

        private string EmitSigned24Product(
            Gen5ShaderInstruction instruction,
            bool high)
        {
            var left = Temp(
                "int",
                $"(as_type<int>(({RawSource(instruction, 0)}) << 8u) >> 8)");
            var right = Temp(
                "int",
                $"(as_type<int>(({RawSource(instruction, 1)}) << 8u) >> 8)");
            var product = Temp("long", $"long({left}) * long({right})");
            return high
                ? $"(uint)({product} >> 32)"
                : $"(uint){product}";
        }

        private string EmitSigned24Mad(Gen5ShaderInstruction instruction)
        {
            var left = Temp(
                "int",
                $"(as_type<int>(({RawSource(instruction, 0)}) << 8u) >> 8)");
            var right = Temp(
                "int",
                $"(as_type<int>(({RawSource(instruction, 1)}) << 8u) >> 8)");
            var product = Temp("long", $"long({left}) * long({right})");
            return $"(uint)({product} + long(as_type<int>({RawSource(instruction, 2)})))";
        }

        private string EmitLerpU8(Gen5ShaderInstruction instruction)
        {
            var first = RawSource(instruction, 0);
            var second = RawSource(instruction, 1);
            var rounding = RawSource(instruction, 2);
            string ByteAverage(uint shift) =>
                $"(((({first} >> {shift}u) & 0xFFu) + " +
                $"(({second} >> {shift}u) & 0xFFu) + " +
                $"(({rounding} >> {shift}u) & 1u)) >> 1u) << {shift}u)";

            return $"({ByteAverage(0)} | {ByteAverage(8)} | " +
                $"{ByteAverage(16)} | {ByteAverage(24)})";
        }

        private string EmitMixedWidthMad16(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5Vop3Control control)
            {
                throw new NotSupportedException($"missing VOP3 control for {instruction.Opcode}");
            }

            var left = EmitVop3HalfBits(instruction, control, 0);
            var right = EmitVop3HalfBits(instruction, control, 1);
            var addend = RawSource(instruction, 2);
            return instruction.Opcode == "VMadI32I16"
                ? $"(uint)((int)(short)({left}) * (int)(short)({right}) + as_type<int>({addend}))"
                : $"((uint)({left}) * (uint)({right}) + ({addend}))";
        }

        private string EmitDivFixupF16(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            if (instruction.Control is not Gen5Vop3Control control)
            {
                throw new NotSupportedException("missing VOP3 control for VDivFixupF16");
            }

            string SourceBits(int index)
            {
                var bits = EmitVop3HalfBits(instruction, control, index);
                if ((control.AbsoluteMask & (1u << index)) != 0)
                {
                    bits = $"(({bits}) & 0x7FFFu)";
                }

                if ((control.NegateMask & (1u << index)) != 0)
                {
                    bits = $"(({bits}) ^ 0x8000u)";
                }

                return Temp("uint", bits);
            }

            var quotient = SourceBits(0);
            var denominator = SourceBits(1);
            var numerator = SourceBits(2);
            var denominatorAbs = Temp("uint", $"{denominator} & 0x7FFFu");
            var numeratorAbs = Temp("uint", $"{numerator} & 0x7FFFu");
            var sign = Temp("uint", $"({denominator} ^ {numerator}) & 0x8000u");
            var denominatorNan = Temp(
                "bool",
                $"(({denominator} & 0x7C00u) == 0x7C00u) && (({denominator} & 0x03FFu) != 0u)");
            var numeratorNan = Temp(
                "bool",
                $"(({numerator} & 0x7C00u) == 0x7C00u) && (({numerator} & 0x03FFu) != 0u)");
            var invalid = Temp(
                "bool",
                $"(({denominatorAbs} == 0u) && ({numeratorAbs} == 0u)) || " +
                $"(({denominatorAbs} == 0x7C00u) && ({numeratorAbs} == 0x7C00u))");
            var infinityResult = Temp(
                "bool",
                $"({denominatorAbs} == 0u) || ({numeratorAbs} == 0x7C00u)");
            var zeroResult = Temp(
                "bool",
                $"({denominatorAbs} == 0x7C00u) || ({numeratorAbs} == 0u)");
            var fixedBits = Temp(
                "uint",
                $"{numeratorNan} ? ({numerator} | 0x0200u) : " +
                $"({denominatorNan} ? ({denominator} | 0x0200u) : " +
                $"({invalid} ? 0xFE00u : " +
                $"({infinityResult} ? ({sign} | 0x7C00u) : " +
                $"({zeroResult} ? {sign} : ({sign} | ({quotient} & 0x7FFFu))))))");
            return (control.OperandSelect & 8) == 0
                ? $"((v[{destination}] & 0xFFFF0000u) | {fixedBits})"
                : $"((v[{destination}] & 0x0000FFFFu) | ({fixedBits} << 16))";
        }

        private string EmitDivFmasF32(Gen5ShaderInstruction instruction)
        {
            var fused = Temp(
                "float",
                $"fma({F(instruction, 0)}, {F(instruction, 1)}, {F(instruction, 2)})");
            return FloatResult(
                instruction,
                $"vcc ? (({fused}) * 4294967296.0f) : ({fused})");
        }

        private string EmitLegacyFloatMultiply(Gen5ShaderInstruction instruction)
        {
            var left = F(instruction, 0);
            var right = F(instruction, 1);
            var product = Temp("float", $"({left}) * ({right})");
            var zeroProduct = LegacyFloatZeroProduct(left, right);
            return FloatResult(instruction, $"{zeroProduct} ? 0.0f : ({product})");
        }

        private string EmitMullitF32(Gen5ShaderInstruction instruction)
        {
            var left = F(instruction, 0);
            var right = F(instruction, 1);
            var product = Temp("float", $"({left}) * ({right})");
            var zeroProduct = LegacyFloatZeroProduct(left, right);
            // V_MULLIT_F32 documents 0.0*x = 0.0. Other special values use
            // the target's normal floating-point multiply behavior.
            return FloatResult(instruction, $"{zeroProduct} ? 0.0f : ({product})");
        }

        private string EmitLegacyFloatMultiplyAccumulate(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            var left = F(instruction, 0);
            var right = F(instruction, 1);
            var product = Temp("float", $"({left}) * ({right})");
            var zeroProduct = LegacyFloatZeroProduct(left, right);
            var productOrZero = $"({zeroProduct} ? 0.0f : ({product}))";
            return FloatResult(
                instruction,
                $"({productOrZero} + as_type<float>(v[{destination}]))");
        }

        private string EmitLegacyFloatMad(Gen5ShaderInstruction instruction)
        {
            var left = F(instruction, 0);
            var right = F(instruction, 1);
            var addend = F(instruction, 2);
            var product = Temp("float", $"({left}) * ({right})");
            var zeroProduct = LegacyFloatZeroProduct(left, right);
            return FloatResult(
                instruction,
                $"(({zeroProduct} ? 0.0f : ({product})) + ({addend}))");
        }

        private string LegacyFloatZeroProduct(string left, string right) => Temp(
            "bool",
            $"((as_type<uint>({left}) & 0x7FFFFFFFu) == 0u) || " +
            $"((as_type<uint>({right}) & 0x7FFFFFFFu) == 0u)");

        private string EmitVop3HalfPack(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5Vop3Control control)
            {
                throw new NotSupportedException($"missing VOP3 control for {instruction.Opcode}");
            }

            if (instruction.Opcode == "VPackB32F16")
            {
                return $"(({EmitVop3HalfBits(instruction, control, 0)}) | " +
                    $"(({EmitVop3HalfBits(instruction, control, 1)}) << 16))";
            }

            var left = EmitVop3F16Operand(instruction, control, 0);
            var right = EmitVop3F16Operand(instruction, control, 1);
            return instruction.Opcode == "VCvtPknormI16F16"
                ? $"pack_float_to_snorm2x16(float2((float){left}, (float){right}))"
                : $"pack_float_to_unorm2x16(float2((float){left}, (float){right}))";
        }

        private string EmitPackedF16(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                throw new NotSupportedException($"missing VOP3P control for {instruction.Opcode}");
            }

            var low = EmitPackedF16Lane(instruction, control, highLane: false);
            var high = EmitPackedF16Lane(instruction, control, highLane: true);
            return $"((uint)as_type<ushort>({low}) | ((uint)as_type<ushort>({high}) << 16))";
        }

        private string EmitPackedInteger16(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                throw new NotSupportedException($"missing VOP3P control for {instruction.Opcode}");
            }

            string EmitLane(bool highLane)
            {
                string SourceBits(int sourceIndex)
                {
                    var raw = RawSource(instruction, sourceIndex);
                    var selectMask = highLane ? control.OpSelHiMask : control.OpSelMask;
                    string bits = ((selectMask >> sourceIndex) & 1) == 0
                        ? $"(({raw}) & 0xFFFFu)"
                        : $"((({raw}) >> 16) & 0xFFFFu)";
                    var negateMask = highLane ? control.NegHiMask : control.NegLoMask;
                    if (((negateMask >> sourceIndex) & 1) != 0)
                    {
                        bits = $"((0u - ({bits})) & 0xFFFFu)";
                    }

                    return Temp("uint", bits);
                }

                var source0 = SourceBits(0);
                var source1 = SourceBits(1);
                var source2 = SourceBits(2);
                var signed0 = $"(int)(short)({source0})";
                var signed1 = $"(int)(short)({source1})";
                var signed2 = $"(int)(short)({source2})";
                string value = instruction.Opcode switch
                {
                    "VPkMadI16" => $"({signed0} * {signed1} + {signed2})",
                    "VPkAddI16" => $"({signed0} + {signed1})",
                    "VPkSubI16" => $"({signed0} - {signed1})",
                    "VPkAshrrevI16" => $"({signed1} >> ({source0} & 15u))",
                    "VPkMaxI16" => $"max({signed0}, {signed1})",
                    "VPkMinI16" => $"min({signed0}, {signed1})",
                    "VPkMulLoU16" => $"({source0} * {source1})",
                    "VPkLshlrevB16" => $"({source1} << ({source0} & 15u))",
                    "VPkLshrrevB16" => $"({source1} >> ({source0} & 15u))",
                    "VPkMadU16" => $"({source0} * {source1} + {source2})",
                    "VPkAddU16" => $"({source0} + {source1})",
                    "VPkSubU16" when control.Clamp =>
                        $"({source0} < {source1} ? 0u : {source0} - {source1})",
                    "VPkSubU16" => $"({source0} - {source1})",
                    "VPkMaxU16" => $"max({source0}, {source1})",
                    "VPkMinU16" => $"min({source0}, {source1})",
                    _ => throw new NotSupportedException(
                        $"unsupported packed integer operation {instruction.Opcode}"),
                };

                var signed = instruction.Opcode is
                    "VPkMadI16" or "VPkAddI16" or "VPkSubI16" or
                    "VPkAshrrevI16" or "VPkMaxI16" or "VPkMinI16";
                var saturatingArithmetic = instruction.Opcode is
                    "VPkMadI16" or "VPkAddI16" or "VPkSubI16" or
                    "VPkMulLoU16" or "VPkMadU16" or "VPkAddU16";
                if (control.Clamp && saturatingArithmetic)
                {
                    value = signed
                        ? $"clamp({value}, -32768, 32767)"
                        : $"min((uint)({value}), 65535u)";
                }

                return $"((uint)({value}) & 0xFFFFu)";
            }

            var low = EmitLane(highLane: false);
            var high = EmitLane(highLane: true);
            return $"(({low}) | (({high}) << 16))";
        }

        private string EmitPackedIntegerDot(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                throw new NotSupportedException($"missing VOP3P control for {instruction.Opcode}");
            }

            var signed = instruction.Opcode.Contains("I32I", StringComparison.Ordinal);
            var componentBits = instruction.Opcode.StartsWith("VDot2", StringComparison.Ordinal)
                ? 16
                : instruction.Opcode.StartsWith("VDot4", StringComparison.Ordinal) ? 8 : 4;
            var componentCount = 32 / componentBits;
            var componentMask = (1u << componentBits) - 1;
            var source0 = RawSource(instruction, 0);
            var source1 = RawSource(instruction, 1);
            var source2 = RawSource(instruction, 2);

            string Component(string source, int index)
            {
                var bits = $"((({source}) >> {index * componentBits}) & 0x{componentMask:X}u)";
                if (!signed)
                {
                    return bits;
                }

                return componentBits switch
                {
                    16 => $"(int)(short)({bits})",
                    8 => $"(int)(char)({bits})",
                    _ => $"(((int)({bits} << 28)) >> 28)",
                };
            }

            var terms = new List<string>(componentCount + 1)
            {
                signed ? $"(long)as_type<int>({source2})" : $"(ulong)({source2})",
            };
            for (var index = 0; index < componentCount; index++)
            {
                terms.Add(
                    $"({(signed ? "long" : "ulong")})({Component(source0, index)}) * " +
                    $"({(signed ? "long" : "ulong")})({Component(source1, index)})");
            }

            var total = Temp(signed ? "long" : "ulong", string.Join(" + ", terms));
            if (control.Clamp)
            {
                total = signed
                    ? Temp("long", $"clamp({total}, (long)-2147483648, (long)2147483647)")
                    : Temp("ulong", $"min({total}, 0xFFFFFFFFul)");
            }

            return $"(uint)({total})";
        }

        private string EmitPackedFloatDot(Gen5ShaderInstruction instruction)
        {
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                throw new NotSupportedException($"missing VOP3P control for {instruction.Opcode}");
            }

            var source2Bits = FlushFloat32DenormalBits(
                Temp("uint", RawSource(instruction, 2)));
            string source2 = $"as_type<float>({source2Bits})";
            if ((control.NegHiMask & 4) != 0)
            {
                source2 = $"fabs({source2})";
            }

            if ((control.NegLoMask & 4) != 0)
            {
                source2 = $"(-{source2})";
            }

            source2 = Temp("float", source2);
            var low = Temp(
                "float",
                $"fma(float({EmitPackedF16Operand(instruction, control, 0, highLane: false)}), " +
                $"float({EmitPackedF16Operand(instruction, control, 1, highLane: false)}), {source2})");
            var dot = Temp(
                "float",
                $"fma(float({EmitPackedF16Operand(instruction, control, 0, highLane: true)}), " +
                $"float({EmitPackedF16Operand(instruction, control, 1, highLane: true)}), {low})");
            var resultBits = FlushFloat32DenormalBits(Temp("uint", AsUInt(dot)));
            if (!control.Clamp)
            {
                return resultBits;
            }

            // Ordered comparisons intentionally map NaN and negative values to
            // zero, matching the VOP3P clamp rule used by the SPIR-V backend.
            var value = Temp("float", $"as_type<float>({resultBits})");
            var clamped = Temp(
                "float",
                $"({value} > 0.0f ? ({value} < 1.0f ? {value} : 1.0f) : 0.0f)");
            return AsUInt(clamped);
        }

        private string FlushFloat32DenormalBits(string bits) =>
            Temp(
                "uint",
                $"((({bits} & 0x7F800000u) == 0u && ({bits} & 0x007FFFFFu) != 0u) " +
                $"? ({bits} & 0x80000000u) : {bits})");

        private string EmitPackedF16Lane(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            bool highLane)
        {
            var left = EmitPackedF16Operand(instruction, control, 0, highLane);
            var right = EmitPackedF16Operand(instruction, control, 1, highLane);
            string value = instruction.Opcode switch
            {
                "VPkAddF16" => $"({left} + {right})",
                "VPkMulF16" => $"({left} * {right})",
                "VPkMinF16" => $"fmin({left}, {right})",
                "VPkMaxF16" => $"fmax({left}, {right})",
                "VPkFmaF16" =>
                    $"fma({left}, {right}, {EmitPackedF16Operand(instruction, control, 2, highLane)})",
                _ => left,
            };
            if (control.Clamp)
            {
                value = $"clamp({value}, half(0.0f), half(1.0f))";
            }

            return Temp("half", value);
        }

        private string EmitPackedF16Operand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            int sourceIndex,
            bool highLane)
        {
            var raw = RawSource(instruction, sourceIndex);
            var selectMask = highLane ? control.OpSelHiMask : control.OpSelMask;
            var bits = ((selectMask >> sourceIndex) & 1) == 0
                ? $"(({raw}) & 0xFFFFu)"
                : $"((({raw}) >> 16) & 0xFFFFu)";
            string value = $"as_type<half>((ushort)({bits}))";
            var negateMask = highLane ? control.NegHiMask : control.NegLoMask;
            if (((negateMask >> sourceIndex) & 1) != 0)
            {
                value = $"(-{value})";
            }

            return Temp("half", value);
        }

        private string EmitFmaMix(Gen5ShaderInstruction instruction, uint destination)
        {
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                throw new NotSupportedException($"missing VOP3P control for {instruction.Opcode}");
            }

            var value =
                $"fma({EmitFmaMixOperand(instruction, control, 0)}, " +
                $"{EmitFmaMixOperand(instruction, control, 1)}, " +
                $"{EmitFmaMixOperand(instruction, control, 2)})";
            if (control.Clamp)
            {
                value = $"clamp({value}, 0.0f, 1.0f)";
            }

            var product = Temp("float", value);
            if (instruction.Opcode == "VFmaMixF32")
            {
                return AsUInt(product);
            }

            var halfBits = Temp("uint", $"(uint)as_type<ushort>(half({product}))");
            return instruction.Opcode == "VFmaMixloF16"
                ? $"((v[{destination}] & 0xFFFF0000u) | {halfBits})"
                : $"((v[{destination}] & 0x0000FFFFu) | ({halfBits} << 16))";
        }

        private string EmitFmaMixOperand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            int sourceIndex)
        {
            var source = instruction.Sources[sourceIndex];
            string value;
            if (((control.OpSelHiMask >> sourceIndex) & 1) != 0 &&
                source.Kind is Gen5OperandKind.VectorRegister or Gen5OperandKind.ScalarRegister)
            {
                var raw = RawSource(instruction, sourceIndex);
                var bits = ((control.OpSelMask >> sourceIndex) & 1) == 0
                    ? $"(({raw}) & 0xFFFFu)"
                    : $"((({raw}) >> 16) & 0xFFFFu)";
                value = $"float(as_type<half>((ushort)({bits})))";
            }
            else
            {
                value = AsFloat(RawSource(instruction, sourceIndex));
            }

            if (((control.NegHiMask >> sourceIndex) & 1) != 0)
            {
                value = $"fabs({value})";
            }

            if (((control.NegLoMask >> sourceIndex) & 1) != 0)
            {
                value = $"(-{value})";
            }

            return Temp("float", value);
        }

        // ---- DPP / SDWA machinery ----

        private static bool IsSupportedDppControl(uint control) =>
            control <= 0xFF ||
            control is >= 0x101 and <= 0x10F or
                >= 0x111 and <= 0x11F or
                >= 0x121 and <= 0x12F or
                0x140 or 0x141 or
                >= 0x150 and <= 0x15F or
                >= 0x160 and <= 0x16F;

        /// <summary>Target lane + in-range flag for a DPP16 control.</summary>
        private (string TargetLane, string InRange) EmitDppSourceLane(Gen5DppControl control)
        {
            var dpp = control.Control;
            if (dpp <= 0xFF)
            {
                // Quad permute: two selector bits per lane-in-quad.
                var selected = Temp(
                    "uint",
                    $"({dpp}u >> ((sharpemu_lane & 3u) * 2u)) & 3u");
                return (Temp("uint", $"(sharpemu_lane & 0xFFFFFFFCu) + {selected}"), "true");
            }

            if (dpp is >= 0x101 and <= 0x10F)
            {
                // row_shl
                var shifted = Temp("uint", $"(sharpemu_lane & 15u) + {dpp & 15}u");
                var inRange = Temp("bool", $"{shifted} < 16u");
                return (Temp("uint", $"(sharpemu_lane & 0xFFFFFFF0u) + ({shifted} & 15u)"), inRange);
            }

            if (dpp is >= 0x111 and <= 0x11F)
            {
                // row_shr
                var inRange = Temp("bool", $"(sharpemu_lane & 15u) >= {dpp & 15}u");
                return (
                    Temp("uint", $"(sharpemu_lane & 0xFFFFFFF0u) + (((sharpemu_lane & 15u) - {dpp & 15}u) & 15u)"),
                    inRange);
            }

            if (dpp is >= 0x121 and <= 0x12F)
            {
                // row_ror
                return (
                    Temp("uint", $"(sharpemu_lane & 0xFFFFFFF0u) + (((sharpemu_lane & 15u) - {dpp & 15}u) & 15u)"),
                    "true");
            }

            var target = dpp switch
            {
                0x140 => "(sharpemu_lane & 0xFFFFFFF0u) + (15u - (sharpemu_lane & 15u))",
                0x141 => "(sharpemu_lane & 0xFFFFFFF8u) + (7u - (sharpemu_lane & 7u))",
                >= 0x150 and <= 0x15F => $"(sharpemu_lane & 0xFFFFFFF0u) + {dpp & 15}u",
                >= 0x160 and <= 0x16F => $"(sharpemu_lane & 0xFFFFFFF0u) + ((sharpemu_lane & 15u) ^ {dpp & 15}u)",
                _ => "sharpemu_lane",
            };
            return (Temp("uint", target), "true");
        }

        // Under the single-lane graphics model every shuffle-select resolves
        // to the lane's own value (the register conceptually holds this
        // thread's value in every lane); compute lanes are real simdgroup
        // threads and shuffle for real. Mirrors the SPIR-V translator's
        // no-subgroup fallback for graphics stages.
        private bool IsSingleLaneStage => _stage != Gen5MslStage.Compute;

        private string ShuffleLane(string value, string targetLane) =>
            IsSingleLaneStage ? value : $"simd_shuffle({value}, (ushort){targetLane})";

        private string LaneActiveExpression(string targetLane) =>
            IsSingleLaneStage ? "exec" : $"simd_shuffle(exec ? 1u : 0u, (ushort){targetLane}) != 0u";

        private string ApplyDppSource(Gen5DppControl control, string value)
        {
            var stored = Temp("uint", value);
            var (targetLane, inRange) = EmitDppSourceLane(control);
            var safeTarget = Temp("uint", $"(({inRange}) ? {targetLane} : sharpemu_lane) & 31u");
            var shuffled = Temp("uint", ShuffleLane(stored, safeTarget));
            if (control.FetchInactive)
            {
                return shuffled;
            }

            var sourceActive = Temp("bool", LaneActiveExpression(safeTarget));
            return Temp("uint", $"(({inRange}) && {sourceActive}) ? {shuffled} : 0u");
        }

        private string ApplyDpp8Source(Gen5Dpp8Control control, string value)
        {
            var stored = Temp("uint", value);
            var selector = Temp(
                "uint",
                $"({control.LaneSelectors}u >> ((sharpemu_lane & 7u) * 3u)) & 7u");
            var targetLane = Temp("uint", $"((sharpemu_lane & 0xFFFFFFF8u) + {selector}) & 31u");
            var shuffled = Temp("uint", ShuffleLane(stored, targetLane));
            if (control.FetchInactive)
            {
                return shuffled;
            }

            var sourceActive = Temp("bool", LaneActiveExpression(targetLane));
            return Temp("uint", $"{sourceActive} ? {shuffled} : 0u");
        }

        private string EmitDppWriteEnabled(Gen5DppControl control)
        {
            var (_, inRange) = EmitDppSourceLane(control);
            var rowEnabled = $"(({control.RowMask}u >> (sharpemu_lane >> 4)) & 1u) != 0u";
            // RDNA2 BANK_MASK partitions each 16-lane row into four contiguous
            // four-lane banks: [0:3], [4:7], [8:11], [12:15]. It does not select
            // the lane's position within each bank.
            var bankEnabled = $"(({control.BankMask}u >> ((sharpemu_lane >> 2u) & 3u)) & 1u) != 0u";
            var sourceAllows = control.BoundControl ? "true" : inRange;
            return Temp("bool", $"({rowEnabled}) && ({bankEnabled}) && ({sourceAllows})");
        }

        private string ApplySdwaDestination(
            Gen5SdwaControl control,
            string value,
            string previous)
        {
            var (shift, width) = control.DestinationSelect switch
            {
                0 => (0u, 8u),
                1 => (8u, 8u),
                2 => (16u, 8u),
                3 => (24u, 8u),
                4 => (0u, 16u),
                5 => (16u, 16u),
                _ => (0u, 32u),
            };
            if (width == 32)
            {
                return value;
            }

            var lowMask = width == 8 ? 0xFFu : 0xFFFFu;
            var fieldMask = lowMask << (int)shift;
            var upperStart = shift + width;
            var upperMask = upperStart == 32 ? 0u : uint.MaxValue << (int)upperStart;
            var positioned = Temp("uint", $"(({value}) & 0x{lowMask:X}u) << {shift}");
            return control.DestinationUnused switch
            {
                // 0: unused bits zeroed. 1: sign-extend upward. 2: preserve.
                0 => positioned,
                1 => Temp(
                    "uint",
                    $"{positioned} | ((({positioned} & 0x{1u << (int)(shift + width - 1):X}u) != 0u) ? 0x{upperMask:X}u : 0u)"),
                2 => Temp("uint", $"(({previous}) & 0x{~fieldMask:X}u) | {positioned}"),
                _ => throw new InvalidOperationException("reserved SDWA destination-unused mode"),
            };
        }

        // ---- compares ----

        private bool TryEmitVectorCompare(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            var opcode = instruction.Opcode;
            string condition;
            if (opcode.EndsWith("F64", StringComparison.Ordinal))
            {
                condition = EmitFloat64Compare(instruction);
            }
            else if (opcode is
                "VCmpClassF32" or "VCmpxClassF32" or
                "VCmpClassF16" or "VCmpxClassF16")
            {
                condition = EmitCompareClass(instruction);
            }
            else if (opcode is
                     "VCmpTruF32" or "VCmpxTruF32" or
                     "VCmpTruF16" or "VCmpxTruF16" or
                     "VCmpTI32" or "VCmpxTI32" or
                     "VCmpTU32" or "VCmpxTU32")
            {
                condition = "true";
            }
            else if (opcode is
                     "VCmpFF32" or "VCmpxFF32" or
                     "VCmpFF16" or "VCmpxFF16" or
                     "VCmpFI32" or "VCmpxFI32" or
                     "VCmpFU32" or "VCmpxFU32")
            {
                condition = "false";
            }
            else if (opcode is
                     "VCmpOF32" or "VCmpxOF32" or
                     "VCmpOF16" or "VCmpxOF16")
            {
                var left = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? EmitScalarF16Operand(instruction, 0)
                    : F(instruction, 0);
                var right = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? EmitScalarF16Operand(instruction, 1)
                    : F(instruction, 1);
                condition = $"(!isnan({left}) && !isnan({right}))";
            }
            else if (opcode is
                      "VCmpUF32" or "VCmpxUF32" or
                     "VCmpUF16" or "VCmpxUF16")
            {
                var left = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? EmitScalarF16Operand(instruction, 0)
                    : F(instruction, 0);
                var right = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? EmitScalarF16Operand(instruction, 1)
                    : F(instruction, 1);
                condition = $"(isnan({left}) || isnan({right}))";
            }
            else if (opcode.EndsWith("I64", StringComparison.Ordinal) ||
                     opcode.EndsWith("U64", StringComparison.Ordinal))
            {
                var signed = opcode.EndsWith("I64", StringComparison.Ordinal);
                var left = Temp(
                    signed ? "long" : "ulong",
                    signed
                        ? $"as_type<long>({RawSource64(instruction, 0)})"
                        : RawSource64(instruction, 0));
                var right = Temp(
                    signed ? "long" : "ulong",
                    signed
                        ? $"as_type<long>({RawSource64(instruction, 1)})"
                        : RawSource64(instruction, 1));
                var op = TrimCompare(opcode) switch
                {
                    "Eq" => "==",
                    "Ne" => "!=",
                    "Lt" => "<",
                    "Le" => "<=",
                    "Gt" => ">",
                    "Ge" => ">=",
                    _ => string.Empty,
                };
                condition = op.Length != 0
                    ? $"({left} {op} {right})"
                    : TrimCompare(opcode) switch
                    {
                        "F" => "false",
                        "T" => "true",
                        _ => string.Empty,
                    };
                if (condition.Length == 0)
                {
                    error = $"unsupported integer 64-bit compare {opcode}";
                    return false;
                }
            }
            else if (opcode.EndsWith("F32", StringComparison.Ordinal) ||
                     opcode.EndsWith("F16", StringComparison.Ordinal))
            {
                // Ordered compares are the plain C operators (false on NaN);
                // the Nxx forms are their unordered negations (true on NaN).
                var (op, unordered) = TrimCompare(opcode) switch
                {
                    "Lt" => ("<", false),
                    "Eq" => ("==", false),
                    "Le" => ("<=", false),
                    "Gt" => (">", false),
                    "Lg" => ("!=", false),
                    "Ge" => (">=", false),
                    "Neq" => ("==", true),
                    "Nlt" => ("<", true),
                    "Nle" => ("<=", true),
                    "Ngt" => (">", true),
                    "Nge" => (">=", true),
                    "Nlg" => ("!=", true),
                    _ => (string.Empty, false),
                };
                if (op.Length == 0)
                {
                    error = $"unsupported float compare {opcode}";
                    return false;
                }

                var left = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? EmitScalarF16Operand(instruction, 0)
                    : F(instruction, 0);
                var right = opcode.EndsWith("F16", StringComparison.Ordinal)
                    ? EmitScalarF16Operand(instruction, 1)
                    : F(instruction, 1);
                var comparison = $"({left} {op} {right})";
                condition = unordered ? $"(!{comparison})" : comparison;
            }
            else
            {
                var is16 = opcode.EndsWith("I16", StringComparison.Ordinal) ||
                           opcode.EndsWith("U16", StringComparison.Ordinal);
                var signed = opcode.EndsWith("I32", StringComparison.Ordinal) ||
                             opcode.EndsWith("I16", StringComparison.Ordinal);
                var op = TrimCompare(opcode) switch
                {
                    "Eq" => "==",
                    "Ne" => "!=",
                    "Lt" => "<",
                    "Le" => "<=",
                    "Gt" => ">",
                    "Ge" => ">=",
                    _ => string.Empty,
                };
                if (op.Length == 0)
                {
                    error = $"unsupported integer compare {opcode}";
                    return false;
                }

                var left = is16 && instruction.Control is Gen5Vop3Control halfControl
                    ? EmitVop3HalfBits(instruction, halfControl, 0)
                    : RawSource(instruction, 0);
                var right = is16 && instruction.Control is Gen5Vop3Control rightHalfControl
                    ? EmitVop3HalfBits(instruction, rightHalfControl, 1)
                    : RawSource(instruction, 1);
                condition = (signed, is16) switch
                {
                    (true, true) => $"((short)(({left}) & 0xFFFFu) {op} (short)(({right}) & 0xFFFFu))",
                    (false, true) => $"((ushort)(({left}) & 0xFFFFu) {op} (ushort)(({right}) & 0xFFFFu))",
                    (true, false) => $"(as_type<int>({left}) {op} as_type<int>({right}))",
                    _ => $"(({left}) {op} ({right}))",
                };
            }

            // Only EXEC-enabled lanes can pass; balloting the raw condition
            // would leak results from disabled lanes into saveexec/branches.
            var active = Temp("bool", $"exec && {condition}");
            if (instruction.Control is Gen5DppControl compareDpp)
            {
                var writeEnabled = EmitDppWriteEnabled(compareDpp);
                active = Temp("bool", $"({writeEnabled}) ? {active} : vcc");
            }

            if (opcode.StartsWith("VCmpx", StringComparison.Ordinal))
            {
                // GFX10 VCMPX writes EXEC only.
                Line($"exec = {active};");
                EmitBallotStore(ExecLoRegister, "exec");
            }
            else
            {
                var target = instruction.Control switch
                {
                    Gen5SdwaControl { ScalarDestination: { } scalarDestination } =>
                        scalarDestination,
                    Gen5Vop3Control { ScalarDestination: { } scalarDestination } =>
                        scalarDestination,
                    _ => VccLoRegister,
                };
                StoreMaskBit(target, active);
            }

            return true;
        }

        private string EmitFloat64Compare(Gen5ShaderInstruction instruction)
        {
            var left = Temp("ulong", Float64SourceBits(instruction, 0));
            var right = Temp("ulong", Float64SourceBits(instruction, 1));
            var leftNan = Temp(
                "bool",
                $"(({left} & 0x7FF0000000000000ul) == 0x7FF0000000000000ul) && " +
                $"(({left} & 0x000FFFFFFFFFFFFFul) != 0ul)");
            var rightNan = Temp(
                "bool",
                $"(({right} & 0x7FF0000000000000ul) == 0x7FF0000000000000ul) && " +
                $"(({right} & 0x000FFFFFFFFFFFFFul) != 0ul)");
            var unordered = Temp("bool", $"{leftNan} || {rightNan}");
            var ordered = Temp("bool", $"!{unordered}");
            var bothZero = Temp(
                "bool",
                $"(({left} & 0x7FFFFFFFFFFFFFFFul) == 0ul) && " +
                $"(({right} & 0x7FFFFFFFFFFFFFFFul) == 0ul)");
            var equal = Temp("bool", $"{ordered} && (({left} == {right}) || {bothZero})");
            var leftKey = Temp(
                "ulong",
                $"(({left} & 0x8000000000000000ul) != 0ul) ? ~{left} : ({left} ^ 0x8000000000000000ul)");
            var rightKey = Temp(
                "ulong",
                $"(({right} & 0x8000000000000000ul) != 0ul) ? ~{right} : ({right} ^ 0x8000000000000000ul)");
            var less = Temp("bool", $"{ordered} && ({leftKey} < {rightKey})");
            var greater = Temp("bool", $"{ordered} && ({leftKey} > {rightKey})");
            var lessEqual = Temp("bool", $"{less} || {equal}");
            var greaterEqual = Temp("bool", $"{greater} || {equal}");
            var lessGreater = Temp("bool", $"{less} || {greater}");
            return TrimCompare(instruction.Opcode) switch
            {
                "F" => "false",
                "Lt" => less,
                "Eq" => equal,
                "Le" => lessEqual,
                "Gt" => greater,
                "Lg" => lessGreater,
                "Ge" => greaterEqual,
                "O" => ordered,
                "U" => unordered,
                "Nge" => $"!{greaterEqual}",
                "Nlg" => $"!{lessGreater}",
                "Ngt" => $"!{greater}",
                "Nle" => $"!{lessEqual}",
                "Neq" => $"!{equal}",
                "Nlt" => $"!{less}",
                "Tru" => "true",
                _ => "false",
            };
        }

        private string Float64SourceBits(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            double? constant = operand.Kind == Gen5OperandKind.EncodedConstant
                ? operand.Value switch
                {
                    125 => 0.0,
                    >= 128 and <= 192 => operand.Value - 128,
                    >= 193 and <= 208 => -(double)(operand.Value - 192),
                    >= 240 and <= 248 when Gen5InlineConstants.TryDecode(
                        operand.Value,
                        out var floatBits) => BitConverter.UInt32BitsToSingle(floatBits),
                    _ => null,
                }
                : null;
            var bits = constant.HasValue
                ? $"0x{BitConverter.DoubleToUInt64Bits(constant.Value):X16}ul"
                : RawSource64(instruction, sourceIndex);
            if (instruction.Control is Gen5Vop3Control control)
            {
                if ((control.AbsoluteMask & (1u << sourceIndex)) != 0)
                {
                    bits = $"(({bits}) & 0x7FFFFFFFFFFFFFFFul)";
                }

                if ((control.NegateMask & (1u << sourceIndex)) != 0)
                {
                    bits = $"(({bits}) ^ 0x8000000000000000ul)";
                }
            }

            return bits;
        }

        private string EmitCompareClass(Gen5ShaderInstruction instruction)
        {
            var half = instruction.Opcode.EndsWith("F16", StringComparison.Ordinal);
            var source = Temp(
                half ? "half" : "float",
                half ? EmitScalarF16Operand(instruction, 0) : F(instruction, 0));
            var raw = Temp(
                "uint",
                half ? EmitScalarF16SourceBits(instruction, 0) : RawSource(instruction, 0));
            var mask = Temp("uint", RawSource(instruction, 1));
            var negative = Temp("bool", $"({raw} & {(half ? "0x8000u" : "0x80000000u")}) != 0u");
            var nan = Temp("bool", $"isnan({source})");
            var infinite = Temp("bool", $"isinf({source})");
            var zero = Temp("bool", $"{source} == {(half ? "half(0.0f)" : "0.0f")}");
            var subnormal = Temp(
                "bool",
                half
                    ? $"fabs({source}) > half(0.0f) && fabs({source}) < as_type<half>((ushort)0x0400)"
                    : $"fabs({source}) > 0.0f && fabs({source}) < as_type<float>(0x00800000u)");
            var normal = Temp(
                "bool",
                $"!({nan} || {infinite} || {zero} || {subnormal})");
            // Class bits: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -subnormal,
            // 5 -zero, 6 +zero, 7 +subnormal, 8 +normal, 9 +inf.
            return Temp(
                "bool",
                $"((({mask} & 3u) != 0u) && {nan}) || " +
                $"((({mask} >> 2) & 1u) != 0u && {infinite} && {negative}) || " +
                $"((({mask} >> 3) & 1u) != 0u && {normal} && {negative}) || " +
                $"((({mask} >> 4) & 1u) != 0u && {subnormal} && {negative}) || " +
                $"((({mask} >> 5) & 1u) != 0u && {zero} && {negative}) || " +
                $"((({mask} >> 6) & 1u) != 0u && {zero} && !{negative}) || " +
                $"((({mask} >> 7) & 1u) != 0u && {subnormal} && !{negative}) || " +
                $"((({mask} >> 8) & 1u) != 0u && {normal} && !{negative}) || " +
                $"((({mask} >> 9) & 1u) != 0u && {infinite} && !{negative})");
        }

        private static string TrimCompare(string opcode)
        {
            var trimmed = opcode.StartsWith("VCmpx", StringComparison.Ordinal)
                ? opcode["VCmpx".Length..]
                : opcode["VCmp".Length..];
            return trimmed[..^3];
        }

        private void StoreCarryOut(Gen5ShaderInstruction instruction, string carryCondition)
        {
            var active = Temp("bool", $"exec && ({carryCondition})");
            var target = instruction.Control is Gen5Vop3Control { ScalarDestination: { } register }
                ? register
                : VccLoRegister;
            StoreMaskBit(target, active);
        }

        /// <summary>
        /// Writes this lane's bit of a wave mask: VCC/EXEC update the per-lane
        /// bool and mirror the ballot into their architectural SGPRs; a plain
        /// SGPR receives the ballot of the per-lane condition.
        /// </summary>
        private void StoreMaskBit(uint register, string condition)
        {
            switch (register)
            {
                case VccLoRegister:
                    Line($"vcc = {condition};");
                    EmitBallotStore(VccLoRegister, "vcc");
                    return;
                case ExecLoRegister:
                    Line($"exec = {condition};");
                    EmitBallotStore(ExecLoRegister, "exec");
                    return;
                default:
                    if (register < ScalarRegisterFileCount)
                    {
                        EmitBallotStore(register, condition);
                    }

                    return;
            }
        }

        /// <summary>Broadcasts <paramref name="value"/> from the first guest-active
        /// lane (lowest set bit of the 64-lane EXEC mask) to all lanes, through the
        /// threadgroup broadcast slot — mirroring the SPIR-V translator's
        /// BroadcastFirstWave64Active. Returns the temp holding the result.</summary>
        private string EmitWave64ReadFirstLane(string value)
        {
            Line("if (sharpemu_lane == 0u) { sharpemu_wave_scratch[2] = 0u; }");
            // 64-lane EXEC mask across both halves (slots 0/1), broadcast in 2.
            Line("sharpemu_wave_scratch[(sharpemu_lane >> 5) & 1u] = sharpemu_ballot(exec);");
            Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            var lo = Temp("uint", "sharpemu_wave_scratch[0]");
            var hi = Temp("uint", "sharpemu_wave_scratch[1]");
            var first = Temp(
                "uint",
                $"({lo} != 0u) ? (uint)ctz({lo}) : (({hi} != 0u) ? (32u + (uint)ctz({hi})) : 0u)");
            var anyActive = Temp("bool", $"(({lo}) | ({hi})) != 0u");
            Line($"if ({anyActive} && sharpemu_lane == {first}) {{ sharpemu_wave_scratch[2] = {value}; }}");
            Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            var result = Temp("uint", "sharpemu_wave_scratch[2]");
            Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            return result;
        }

        /// <summary>Stores the wave ballot of <paramref name="condition"/> into the
        /// mask register pair (low, low+1). Wave32 fills the low dword and clears
        /// the high; wave64 bridges both 32-wide halves through threadgroup
        /// scratch so the pair holds the full 64-lane mask. The bridging barriers
        /// are safe because the guest program's scalar PC keeps all 64 lanes in
        /// lockstep through the dispatcher (one wave per threadgroup).</summary>
        private void EmitBallotStore(uint loRegister, string condition)
        {
            var hiRegister = loRegister + 1;
            if (!IsWave64)
            {
                Line($"s[{loRegister}] = sharpemu_ballot({condition});");
                if (hiRegister < ScalarRegisterFileCount)
                {
                    Line($"s[{hiRegister}] = 0u;");
                }

                return;
            }

            // simd_ballot is uniform across a simdgroup, so every lane of a half
            // writes the same 32-bit value to that half's slot — no first-lane
            // guard needed. Barrier, read both halves, barrier before the slot
            // can be reused by the next ballot.
            Line($"sharpemu_wave_scratch[(sharpemu_lane >> 5) & 1u] = sharpemu_ballot({condition});");
            Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            Line($"s[{loRegister}] = sharpemu_wave_scratch[0];");
            if (hiRegister < ScalarRegisterFileCount)
            {
                Line($"s[{hiRegister}] = sharpemu_wave_scratch[1];");
            }

            Line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        }

        // ---- permlane / cube ----

        private string EmitPermlane16(Gen5ShaderInstruction instruction, bool exchangeRows)
        {
            if (instruction.Control is not Gen5Vop3Control control ||
                (control.OperandSelect & ~3u) != 0 ||
                control.AbsoluteMask != 0 ||
                control.NegateMask != 0 ||
                control.OutputModifier != 0 ||
                control.Clamp)
            {
                throw new NotSupportedException(
                    $"invalid permlane modifiers for {instruction.Opcode}");
            }

            var value = Temp("uint", RawSource(instruction, 0));
            var selectorLow = Temp("uint", RawSource(instruction, 1));
            var selectorHigh = Temp("uint", RawSource(instruction, 2));
            var localLane = Temp("uint", "sharpemu_lane & 15u");
            var selector = Temp(
                "uint",
                $"({localLane} < 8u ? ({selectorLow} >> ({localLane} << 2)) : ({selectorHigh} >> (({localLane} - 8u) << 2))) & 15u");
            var rowBase = exchangeRows
                ? "((sharpemu_lane & 0xFFFFFFF0u) ^ 16u)"
                : "(sharpemu_lane & 0xFFFFFFF0u)";
            var targetLane = Temp("uint", $"({rowBase} + {selector}) & 31u");
            var shuffled = Temp("uint", ShuffleLane(value, targetLane));
            var fetchInactive = (control.OperandSelect & 1) != 0;
            if (fetchInactive)
            {
                return shuffled;
            }

            var sourceActive = Temp("bool", LaneActiveExpression(targetLane));
            return Temp("uint", $"{sourceActive} ? {shuffled} : 0u");
        }

        private enum CubeCoordinate
        {
            Id,
            Sc,
            Tc,
            Ma,
        }

        private string EmitCubeCoordinate(
            Gen5ShaderInstruction instruction,
            CubeCoordinate coordinate)
        {
            var x = Temp("float", F(instruction, 0));
            var y = Temp("float", F(instruction, 1));
            var z = Temp("float", F(instruction, 2));
            var amaxXY = Temp("float", $"fmax(fabs({x}), fabs({y}))");
            var amax = Temp("float", $"fmax(fabs({z}), {amaxXY})");
            if (coordinate == CubeCoordinate.Ma)
            {
                return FloatResult(instruction, $"2.0f * {amax}");
            }

            var isZMax = Temp("bool", $"fabs({z}) >= {amaxXY}");
            var yGeX = Temp("bool", $"fabs({y}) >= fabs({x})");
            var isYMax = Temp("bool", $"!{isZMax} && {yGeX}");
            switch (coordinate)
            {
                case CubeCoordinate.Id:
                {
                    var zCase = $"({z} < 0.0f ? 5.0f : 4.0f)";
                    var yCase = $"({y} < 0.0f ? 3.0f : 2.0f)";
                    var xCase = $"({x} < 0.0f ? 1.0f : 0.0f)";
                    return FloatResult(
                        instruction,
                        $"({isZMax} ? {zCase} : ({yGeX} ? {yCase} : {xCase}))");
                }
                case CubeCoordinate.Sc:
                {
                    var zCase = $"({z} < 0.0f ? (-{x}) : {x})";
                    var xCase = $"({x} < 0.0f ? {z} : (-{z}))";
                    return FloatResult(
                        instruction,
                        $"({isZMax} ? {zCase} : ({isYMax} ? {x} : {xCase}))");
                }
                default:
                {
                    var yCase = $"({y} < 0.0f ? (-{z}) : {z})";
                    return FloatResult(
                        instruction,
                        $"({isYMax} ? {yCase} : (-{y}))");
                }
            }
        }

        // ---- scalar ALU ----

        private bool TryEmitScalarAlu(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Encoding == Gen5ShaderEncoding.Sopc)
            {
                return TryEmitScalarCompare(instruction, out error);
            }

            if (instruction.Destinations.Count == 0 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister)
            {
                error = "missing scalar destination";
                return false;
            }

            var destination = instruction.Destinations[0].Value;
            if (instruction.Encoding == Gen5ShaderEncoding.Sopk)
            {
                var immediate = unchecked((uint)(short)(instruction.Words[0] & 0xFFFF));
                if (instruction.Opcode.StartsWith("SCmpk", StringComparison.Ordinal))
                {
                    return TryEmitScalarCompareK(instruction, destination, immediate, out error);
                }

                var value = instruction.Opcode switch
                {
                    "SMovkI32" => FormatUInt(immediate),
                    "SAddkI32" => $"({ScalarExpression(destination)} + {FormatUInt(immediate)})",
                    "SMulkI32" => $"({ScalarExpression(destination)} * {FormatUInt(immediate)})",
                    _ => string.Empty,
                };
                if (value.Length == 0)
                {
                    error = $"unsupported scalar immediate {instruction.Opcode}";
                    return false;
                }

                StoreScalar(destination, Temp("uint", value));
                return true;
            }

            if (instruction.Opcode == "SGetpcB64")
            {
                var pc = _state.Program.Address +
                    instruction.Pc +
                    (ulong)(instruction.Words.Count * sizeof(uint));
                StoreScalar(destination, FormatUInt((uint)pc));
                StoreScalar(destination + 1, FormatUInt((uint)(pc >> 32)));
                return true;
            }

            if (instruction.Opcode is
                "SMovrelsB32" or "SMovrelsB64" or
                "SMovreldB32" or "SMovreldB64")
            {
                return TryEmitScalarRelativeMove(instruction, destination, out error);
            }

            if (instruction.Opcode.EndsWith("B64", StringComparison.Ordinal) ||
                instruction.Opcode is "SBfeU64" or "SBfeI64" or "SAshrI64")
            {
                return TryEmitScalar64(instruction, destination, out error);
            }

            var left = Temp("uint", RawSource(instruction, 0));
            if (instruction.Opcode.EndsWith("SaveexecB32", StringComparison.Ordinal))
            {
                var oldExec = Temp("uint", $"s[{ExecLoRegister}]");
                var operation = instruction.Opcode[1..instruction.Opcode.IndexOf(
                    "Saveexec",
                    StringComparison.Ordinal)];
                var combined = operation switch
                {
                    "And" => $"({left} & {oldExec})",
                    "Or" => $"({left} | {oldExec})",
                    "Xor" => $"({left} ^ {oldExec})",
                    "Nand" => $"~({left} & {oldExec})",
                    "Nor" => $"~({left} | {oldExec})",
                    "Xnor" => $"~({left} ^ {oldExec})",
                    "Andn1" => $"(~{left} & {oldExec})",
                    "Andn2" => $"({left} & ~{oldExec})",
                    "Orn1" => $"(~{left} | {oldExec})",
                    "Orn2" => $"({left} | ~{oldExec})",
                    _ => string.Empty,
                };
                if (combined.Length == 0)
                {
                    error = $"unsupported scalar 32-bit saveexec opcode {instruction.Opcode}";
                    return false;
                }

                var mask = Temp("uint", combined);
                StoreScalar(destination, oldExec);
                Line($"s[{ExecLoRegister}] = {mask};");
                Line($"s[{ExecHiRegister}] = 0u;");
                Line($"exec = (({mask} >> sharpemu_lane) & 1u) != 0u;");
                Line($"scc = {mask} != 0u;");
                return true;
            }

            switch (instruction.Opcode)
            {
                case "SMovB32":
                    StoreScalar(destination, left);
                    return true;
                case "SNotB32":
                {
                    var result = Temp("uint", $"~{left}");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SBrevB32":
                {
                    var result = Temp("uint", $"reverse_bits({left})");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SBcnt1I32B32":
                {
                    var result = Temp("uint", $"popcount({left})");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SFF1I32B32":
                {
                    var result = Temp(
                        "uint",
                        $"{left} == 0u ? 0xFFFFFFFFu : (uint)ctz({left})");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SBitset1B32":
                    StoreScalar(
                        destination,
                        $"{ScalarExpression(destination)} | (1u << ({left} & 31u))");
                    return true;
            }

            if (instruction.Sources.Count < 2)
            {
                error = $"missing scalar source for {instruction.Opcode}";
                return false;
            }

            var right = Temp("uint", RawSource(instruction, 1));
            string resultExpression;
            string sccStatement;
            switch (instruction.Opcode)
            {
                case "SAddU32":
                    resultExpression = $"({left} + {right})";
                    sccStatement = "RESULT < " + left;
                    break;
                case "SSubU32":
                    resultExpression = $"({left} - {right})";
                    sccStatement = $"{right} > {left}";
                    break;
                case "SAddI32":
                    resultExpression = $"({left} + {right})";
                    sccStatement = $"((~({left} ^ {right}) & ({left} ^ RESULT)) >> 31) != 0u";
                    break;
                case "SSubI32":
                    resultExpression = $"({left} - {right})";
                    sccStatement = $"(((({left} ^ {right})) & ({left} ^ RESULT)) >> 31) != 0u";
                    break;
                case "SAddcU32":
                {
                    var partial = Temp("uint", $"{left} + {right}");
                    var sum = Temp("uint", $"{partial} + (scc ? 1u : 0u)");
                    Line($"scc = ({partial} < {left}) || ({sum} < {partial});");
                    StoreScalar(destination, sum);
                    return true;
                }
                case "SSubbU32":
                {
                    var borrow = Temp("uint", "scc ? 1u : 0u");
                    var partial = Temp("uint", $"{left} - {right}");
                    var difference = Temp("uint", $"{partial} - {borrow}");
                    Line($"scc = ({right} > {left}) || (({borrow} == 1u) && ({right} == {left}));");
                    StoreScalar(destination, difference);
                    return true;
                }
                case "SMulI32":
                    resultExpression = $"({left} * {right})";
                    sccStatement = string.Empty;
                    break;
                case "SMulHiU32":
                    resultExpression = $"mulhi({left}, {right})";
                    sccStatement = string.Empty;
                    break;
                case "SMulHiI32":
                    resultExpression =
                        $"as_type<uint>((int)(((long)as_type<int>({left}) * (long)as_type<int>({right})) >> 32))";
                    sccStatement = string.Empty;
                    break;
                case "SAndB32":
                    resultExpression = $"({left} & {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SOrB32":
                    resultExpression = $"({left} | {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SXorB32":
                    resultExpression = $"({left} ^ {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SNandB32":
                    resultExpression = $"~({left} & {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SNorB32":
                    resultExpression = $"~({left} | {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SXnorB32":
                    resultExpression = $"~({left} ^ {right})";
                    sccStatement = "NONZERO";
                    break;
                case "SAndn2B32":
                    resultExpression = $"({left} & ~{right})";
                    sccStatement = "NONZERO";
                    break;
                case "SOrn2B32":
                    resultExpression = $"({left} | ~{right})";
                    sccStatement = "NONZERO";
                    break;
                case "SLshlB32":
                    resultExpression = $"({left} << ({right} & 31u))";
                    sccStatement = "NONZERO";
                    break;
                case "SLshrB32":
                    resultExpression = $"({left} >> ({right} & 31u))";
                    sccStatement = "NONZERO";
                    break;
                case "SAshrI32":
                    resultExpression = $"(uint)(as_type<int>({left}) >> ({right} & 31u))";
                    sccStatement = "NONZERO";
                    break;
                case "SAbsdiffI32":
                {
                    var difference = Temp("uint", $"{left} - {right}");
                    resultExpression =
                        $"(({difference} & 0x80000000u) != 0u ? (0u - {difference}) : {difference})";
                    sccStatement = "NONZERO";
                    break;
                }
                case "SBfmB32":
                    resultExpression = $"(((1u << ({left} & 31u)) - 1u) << ({right} & 31u))";
                    sccStatement = string.Empty;
                    break;
                case "SBfeU32":
                case "SBfeI32":
                {
                    // Width clamps to the bits remaining above the offset.
                    var offset = Temp("uint", $"{right} & 31u");
                    var width = Temp(
                        "uint",
                        $"min(({right} >> 16) & 0x7Fu, 32u - {offset})");
                    var result = instruction.Opcode == "SBfeI32"
                        ? Temp(
                            "uint",
                            $"{width} == 0u ? 0u : (uint)extract_bits(as_type<int>({left}), {offset}, {width})")
                        : Temp(
                            "uint",
                            $"{width} == 0u ? 0u : extract_bits({left}, {offset}, {width})");
                    StoreScalar(destination, result);
                    Line($"scc = {result} != 0u;");
                    return true;
                }
                case "SCselectB32":
                    resultExpression = $"(scc ? {left} : {right})";
                    sccStatement = string.Empty;
                    break;
                case "SMinU32":
                    resultExpression = $"min({left}, {right})";
                    sccStatement = $"{left} < {right}";
                    break;
                case "SMaxU32":
                    resultExpression = $"max({left}, {right})";
                    sccStatement = $"{left} > {right}";
                    break;
                case "SMinI32":
                    resultExpression = $"(uint)min(as_type<int>({left}), as_type<int>({right}))";
                    sccStatement = $"as_type<int>({left}) < as_type<int>({right})";
                    break;
                case "SMaxI32":
                    resultExpression = $"(uint)max(as_type<int>({left}), as_type<int>({right}))";
                    sccStatement = $"as_type<int>({left}) > as_type<int>({right})";
                    break;
                case "SLshl1AddU32":
                case "SLshl2AddU32":
                case "SLshl3AddU32":
                case "SLshl4AddU32":
                {
                    var shift = (uint)(instruction.Opcode[5] - '0');
                    resultExpression = $"(({left} << {shift}) + {right})";
                    sccStatement = string.Empty;
                    break;
                }
                case "SPackLlB32B16":
                    resultExpression = $"(({left} & 0xFFFFu) | ({right} << 16))";
                    sccStatement = string.Empty;
                    break;
                case "SPackLhB32B16":
                    resultExpression = $"(({left} & 0xFFFFu) | ({right} & 0xFFFF0000u))";
                    sccStatement = string.Empty;
                    break;
                case "SPackHhB32B16":
                    resultExpression = $"(({left} >> 16) | ({right} & 0xFFFF0000u))";
                    sccStatement = string.Empty;
                    break;
                default:
                    error = $"unsupported scalar opcode {instruction.Opcode}";
                    return false;
            }

            var value2 = Temp("uint", resultExpression);
            StoreScalar(destination, value2);
            if (sccStatement == "NONZERO")
            {
                Line($"scc = {value2} != 0u;");
            }
            else if (sccStatement.Length != 0)
            {
                Line($"scc = {sccStatement.Replace("RESULT", value2)};");
            }

            return true;
        }

        private bool TryEmitScalarCompare(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count < 2)
            {
                error = "missing scalar compare source";
                return false;
            }

            var left = Temp("uint", RawSource(instruction, 0));
            var right = Temp("uint", RawSource(instruction, 1));
            if (instruction.Opcode is "SBitcmp0B32" or "SBitcmp1B32")
            {
                var isSet = $"(({left} >> ({right} & 31u)) & 1u) != 0u";
                Line(instruction.Opcode == "SBitcmp1B32"
                    ? $"scc = {isSet};"
                    : $"scc = !({isSet});");
                return true;
            }

            if (instruction.Opcode is "SBitcmp0B64" or "SBitcmp1B64")
            {
                var wideLeft = Temp("ulong", RawSource64(instruction, 0));
                var isSet = $"(({wideLeft} >> (ulong({right}) & 63ul)) & 1ul) != 0ul";
                Line(instruction.Opcode == "SBitcmp1B64"
                    ? $"scc = {isSet};"
                    : $"scc = !({isSet});");
                return true;
            }

            return TryEmitScalarCompareCore(instruction.Opcode, "SCmp", left, right, out error);
        }

        private bool TryEmitScalarCompareK(
            Gen5ShaderInstruction instruction,
            uint destination,
            uint immediate,
            out string error) =>
            TryEmitScalarCompareCore(
                instruction.Opcode,
                "SCmpk",
                ScalarExpression(destination),
                FormatUInt(immediate),
                out error);

        private bool TryEmitScalarCompareCore(
            string opcode,
            string prefix,
            string left,
            string right,
            out string error)
        {
            error = string.Empty;
            var suffix = opcode[prefix.Length..];
            var signed = suffix.EndsWith("I32", StringComparison.Ordinal);
            var op = suffix[..^3] switch
            {
                "Eq" => "==",
                "Lg" => "!=",
                "Gt" => ">",
                "Ge" => ">=",
                "Lt" => "<",
                "Le" => "<=",
                _ => string.Empty,
            };
            if (op.Length == 0)
            {
                error = $"unsupported scalar compare {opcode}";
                return false;
            }

            Line(signed
                ? $"scc = as_type<int>({left}) {op} as_type<int>({right});"
                : $"scc = ({left}) {op} ({right});");
            return true;
        }

        private bool TryEmitScalarRelativeMove(
            Gen5ShaderInstruction instruction,
            uint destination,
            out string error)
        {
            error = string.Empty;
            if (instruction.Sources.Count != 1 ||
                instruction.Sources[0].Kind != Gen5OperandKind.ScalarRegister)
            {
                error = $"{instruction.Opcode} expects an SGPR source base";
                return false;
            }

            const uint m0Register = 124;
            var m0 = Temp("uint", $"s[{m0Register}]");
            var relativeSource = instruction.Opcode.StartsWith(
                "SMovrels",
                StringComparison.Ordinal);
            var is64 = instruction.Opcode.EndsWith("B64", StringComparison.Ordinal);
            var source = instruction.Sources[0].Value;

            var low = relativeSource
                ? LoadScalarRelative(source, m0)
                : Temp("uint", ScalarExpression(source));
            var high = is64
                ? relativeSource
                    ? LoadScalarRelative(source + 1, m0)
                    : Temp("uint", ScalarExpression(source + 1))
                : string.Empty;

            if (relativeSource)
            {
                StoreScalar(destination, low);
                if (is64)
                {
                    StoreScalar(destination + 1, high);
                }
            }
            else
            {
                StoreScalarRelative(destination, m0, low);
                if (is64)
                {
                    StoreScalarRelative(destination + 1, m0, high);
                }
            }

            return true;
        }

        // ---- 64-bit scalar ops over register pairs ----

        private bool TryEmitScalar64(
            Gen5ShaderInstruction instruction,
            uint destination,
            out string error)
        {
            error = string.Empty;
            var left = Temp("ulong", RawSource64(instruction, 0));
            if (instruction.Opcode.EndsWith("SaveexecB64", StringComparison.Ordinal))
            {
                var oldExec = Temp("ulong", Scalar64Expression(ExecLoRegister));
                var operation = instruction.Opcode[1..instruction.Opcode.IndexOf(
                    "Saveexec",
                    StringComparison.Ordinal)];
                var combined = operation switch
                {
                    "And" => $"({left} & {oldExec})",
                    "Or" => $"({left} | {oldExec})",
                    "Xor" => $"({left} ^ {oldExec})",
                    "Nand" => $"~({left} & {oldExec})",
                    "Nor" => $"~({left} | {oldExec})",
                    "Xnor" => $"~({left} ^ {oldExec})",
                    "Andn1" => $"(~{left} & {oldExec})",
                    "Andn2" => $"({left} & ~{oldExec})",
                    "Orn1" => $"(~{left} | {oldExec})",
                    "Orn2" => $"({left} | ~{oldExec})",
                    _ => string.Empty,
                };
                if (combined.Length == 0)
                {
                    error = $"unsupported scalar 64-bit saveexec opcode {instruction.Opcode}";
                    return false;
                }

                var mask = Temp("ulong", combined);
                StoreScalar64(destination, oldExec);
                Line($"s[{ExecLoRegister}] = (uint){mask};");
                Line($"s[{ExecHiRegister}] = (uint)({mask} >> 32);");
                Line($"exec = ((((uint){mask}) >> sharpemu_lane) & 1u) != 0u;");
                Line($"scc = {mask} != 0ul;");
                return true;
            }

            string value;
            var setsScc = true;
            switch (instruction.Opcode)
            {
                case "SMovB64":
                    value = left;
                    setsScc = false;
                    break;
                case "SNotB64":
                    value = $"~{left}";
                    break;
                case "SWqmB64":
                {
                    // Whole-quad mode: each 4-lane group becomes all-ones if any
                    // of its bits is set.
                    var quadAny = Temp(
                        "ulong",
                        $"({left} | ({left} >> 1) | ({left} >> 2) | ({left} >> 3)) & 0x1111111111111111ul");
                    value = $"({quadAny} * 0xFul)";
                    break;
                }
                case "SLshlB64" or "SLshrB64" or "SAshrI64":
                {
                    var shift = Temp("uint", $"({RawSource(instruction, 1)}) & 63u");
                    value = instruction.Opcode switch
                    {
                        "SLshlB64" => $"({left} << {shift})",
                        "SLshrB64" => $"({left} >> {shift})",
                        _ => $"as_type<ulong>(as_type<long>({left}) >> {shift})",
                    };
                    break;
                }
                case "SBfmB64":
                {
                    var width = Temp("ulong", $"(ulong)(({RawSource(instruction, 0)}) & 63u)");
                    var offset = Temp("ulong", $"(ulong)(({RawSource(instruction, 1)}) & 63u)");
                    value = $"((((1ul << {width}) - 1ul)) << {offset})";
                    break;
                }
                case "SBfeU64" or "SBfeI64":
                {
                    var control = Temp("uint", RawSource(instruction, 1));
                    var offset = Temp("uint", $"{control} & 63u");
                    var width = Temp("uint", $"min(({control} >> 16) & 0x7Fu, 64u - {offset})");
                    var mask = Temp(
                        "ulong",
                        $"{width} >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << {width}) - 1ul)");
                    var extracted = Temp("ulong", $"({left} >> {offset}) & {mask}");
                    if (instruction.Opcode == "SBfeI64")
                    {
                        var signBit = Temp(
                            "ulong",
                            $"{width} == 0u ? 0ul : (1ul << ({width} - 1u))");
                        extracted = Temp(
                            "ulong",
                            $"{width} == 0u ? 0ul : (({extracted} ^ {signBit}) - {signBit})");
                    }

                    value = extracted;
                    break;
                }
                default:
                {
                    if (instruction.Sources.Count < 2)
                    {
                        error = "missing scalar 64-bit source";
                        return false;
                    }

                    var right = Temp("ulong", RawSource64(instruction, 1));
                    value = instruction.Opcode switch
                    {
                        "SAndB64" => $"({left} & {right})",
                        "SOrB64" => $"({left} | {right})",
                        "SXorB64" => $"({left} ^ {right})",
                        "SNandB64" => $"~({left} & {right})",
                        "SNorB64" => $"~({left} | {right})",
                        "SXnorB64" => $"~({left} ^ {right})",
                        "SAndn1B64" => $"(~{left} & {right})",
                        "SAndn2B64" => $"({left} & ~{right})",
                        "SOrn1B64" => $"(~{left} | {right})",
                        "SOrn2B64" => $"({left} | ~{right})",
                        "SCselectB64" => $"(scc ? {left} : {right})",
                        _ => string.Empty,
                    };
                    if (value.Length == 0)
                    {
                        error = $"unsupported scalar 64-bit opcode {instruction.Opcode}";
                        return false;
                    }

                    setsScc = instruction.Opcode != "SCselectB64";
                    break;
                }
            }

            var stored = Temp("ulong", value);
            StoreScalar64(destination, stored);
            if (setsScc)
            {
                Line($"scc = {stored} != 0ul;");
            }

            return true;
        }

        // ---- operand helpers ----

        private uint DestinationVector(Gen5ShaderInstruction instruction)
        {
            var destination = instruction.Destinations[0];
            return destination.Kind == Gen5OperandKind.VectorRegister
                ? destination.Value
                : throw new NotSupportedException(
                    $"vector destination expected in {instruction.Opcode}");
        }

        /// <summary>
        /// Raw 32-bit source with DPP/DPP8 lane remapping on src0 and SDWA
        /// byte/word selection + integer modifiers, mirroring GetRawSource.
        /// </summary>
        private string RawSource(
            Gen5ShaderInstruction instruction,
            int sourceIndex,
            bool applySdwaIntegerModifiers = true)
        {
            var value = SourceExpression(instruction.Sources[sourceIndex], instruction);
            if (sourceIndex == 0 && instruction.Control is Gen5DppControl dpp)
            {
                value = ApplyDppSource(dpp, value);
            }
            else if (sourceIndex == 0 && instruction.Control is Gen5Dpp8Control dpp8)
            {
                value = ApplyDpp8Source(dpp8, value);
            }

            if (instruction.Control is Gen5SdwaControl sdwa)
            {
                var selector = sourceIndex switch
                {
                    0 => sdwa.Source0Select,
                    1 => sdwa.Source1Select,
                    _ => 6u,
                };
                value = selector switch
                {
                    0 => $"(({value}) & 0xFFu)",
                    1 => $"((({value}) >> 8) & 0xFFu)",
                    2 => $"((({value}) >> 16) & 0xFFu)",
                    3 => $"((({value}) >> 24) & 0xFFu)",
                    4 => $"(({value}) & 0xFFFFu)",
                    5 => $"((({value}) >> 16) & 0xFFFFu)",
                    _ => value,
                };
                var signExtend = sourceIndex switch
                {
                    0 => sdwa.Source0SignExtend,
                    1 => sdwa.Source1SignExtend,
                    _ => false,
                };
                if (signExtend && selector != 6)
                {
                    var width = selector <= 3 ? 8u : 16u;
                    value = $"(uint)extract_bits(as_type<int>({value}), 0u, {width}u)";
                }

                if (applySdwaIntegerModifiers)
                {
                    if ((sdwa.AbsoluteMask & (1u << sourceIndex)) != 0)
                    {
                        value = $"(uint)abs(as_type<int>({value}))";
                    }

                    if ((sdwa.NegateMask & (1u << sourceIndex)) != 0)
                    {
                        value = $"(0u - ({value}))";
                    }
                }
            }

            return value;
        }

        /// <summary>64-bit source: SGPR/VGPR pair, sign-extended inline, or zero-extended 32-bit.</summary>
        private string RawSource64(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            switch (operand.Kind)
            {
                case Gen5OperandKind.ScalarRegister:
                    return Scalar64Expression(operand.Value);
                case Gen5OperandKind.VectorRegister:
                    return $"((ulong)v[{operand.Value}] | ((ulong)v[{operand.Value + 1}] << 32))";
                case Gen5OperandKind.EncodedConstant when operand.Value is >= 193 and <= 208:
                {
                    // Inline negatives sign-extend: -1 denotes a full 64-bit mask.
                    var signed = -(long)(operand.Value - 192);
                    return $"0x{unchecked((ulong)signed):X}ul";
                }
                default:
                    return $"(ulong)({RawSource(instruction, sourceIndex)})";
            }
        }

        private string Scalar64Expression(uint register) => register switch
        {
            // VCC/EXEC read their architectural SGPR pairs like any other
            // register — programs park plain data there (see StoreScalar).
            _ when register + 1 < ScalarRegisterFileCount =>
                $"((ulong)s[{register}] | ((ulong)s[{register + 1}] << 32))",
            _ => "0ul",
        };

        private void StoreScalar64(uint register, string ulongValue)
        {
            StoreScalar(register, $"(uint)({ulongValue})");
            StoreScalar(register + 1, $"(uint)(({ulongValue}) >> 32)");
        }

        /// <summary>Float view of a source with abs/neg modifiers from VOP3/SDWA/DPP.</summary>
        private string F(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var expression = AsFloat(
                RawSource(instruction, sourceIndex, applySdwaIntegerModifiers: false));
            var (absoluteMask, negateMask) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.AbsoluteMask, control.NegateMask),
                Gen5SdwaControl control => (control.AbsoluteMask, control.NegateMask),
                Gen5DppControl control => (control.AbsoluteMask, control.NegateMask),
                _ => (0u, 0u),
            };
            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                expression = $"fabs({expression})";
            }

            if ((negateMask & (1u << sourceIndex)) != 0)
            {
                expression = $"(-{expression})";
            }

            return expression;
        }

        /// <summary>Reads the selected 16-bit half as a widened float.</summary>
        private string F16(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            string expression;
            if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                Gen5InlineConstants.TryDecode(operand.Value, out var inline))
            {
                expression = operand.Value switch
                {
                    >= 128 and <= 192 => $"{operand.Value - 128}.0f",
                    >= 193 and <= 208 => $"(-{operand.Value - 192}.0f)",
                    _ => AsFloat(FormatUInt(inline)),
                };
            }
            else
            {
                var raw = RawSource(
                    instruction,
                    sourceIndex,
                    applySdwaIntegerModifiers: false);
                var shift = instruction.Control is Gen5Vop3Control control &&
                    (control.OperandSelect & (1u << sourceIndex)) != 0
                        ? 16
                        : 0;
                expression =
                    $"(float)as_type<half>((ushort)((({raw}) >> {shift}) & 0xFFFFu))";
            }

            var (absoluteMask, negateMask) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.AbsoluteMask, control.NegateMask),
                Gen5SdwaControl control => (control.AbsoluteMask, control.NegateMask),
                Gen5DppControl control => (control.AbsoluteMask, control.NegateMask),
                _ => (0u, 0u),
            };
            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                expression = $"fabs({expression})";
            }

            if ((negateMask & (1u << sourceIndex)) != 0)
            {
                expression = $"(-{expression})";
            }

            return expression;
        }

        /// <summary>Rounds to f16 and preserves the unselected VGPR half.</summary>
        private string Float16Result(
            Gen5ShaderInstruction instruction,
            uint destination,
            string expression)
        {
            var control = instruction.Control as Gen5Vop3Control;
            expression = (control?.OutputModifier ?? 0) switch
            {
                1 => $"(({expression}) * 2.0f)",
                2 => $"(({expression}) * 4.0f)",
                3 => $"(({expression}) * 0.5f)",
                _ => expression,
            };
            if (control?.Clamp == true)
            {
                expression = $"clamp({expression}, 0.0f, 1.0f)";
            }

            var packed = $"(uint)as_type<ushort>(half({expression}))";
            return ((control?.OperandSelect ?? 0) & 8) != 0
                ? $"((v[{destination}] & 0x0000FFFFu) | (({packed}) << 16))"
                : $"((v[{destination}] & 0xFFFF0000u) | ({packed}))";
        }

        /// <summary>
        /// Wraps a float expression with VOP3/SDWA output modifiers and clamp,
        /// then bitcasts back to the register file's uint domain.
        /// </summary>
        private string FloatResult(Gen5ShaderInstruction instruction, string expression)
        {
            var (outputModifier, clamp) = instruction.Control switch
            {
                Gen5Vop3Control control => (control.OutputModifier, control.Clamp),
                Gen5SdwaControl control => (control.OutputModifier, control.Clamp),
                _ => (0u, false),
            };
            expression = outputModifier switch
            {
                1 => $"(({expression}) * 2.0f)",
                2 => $"(({expression}) * 4.0f)",
                3 => $"(({expression}) * 0.5f)",
                _ => expression,
            };
            if (clamp)
            {
                expression = $"clamp({expression}, 0.0f, 1.0f)";
            }

            return AsUInt($"({expression})");
        }

        /// <summary>The lane's bit of a mask operand (VCC/EXEC/SGPR mask).</summary>
        private string MaskBitExpression(Gen5Operand operand) => operand switch
        {
            { Kind: Gen5OperandKind.ScalarRegister, Value: VccLoRegister } => "vcc",
            { Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister } => "exec",
            { Kind: Gen5OperandKind.ScalarRegister } scalar =>
                $"(((s[{scalar.Value}] >> sharpemu_lane) & 1u) != 0u)",
            _ => throw new NotSupportedException("mask operand must be a scalar register"),
        };
    }
}
