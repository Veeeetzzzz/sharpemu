// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Vulkan;

public static partial class Gen5SpirvTranslator
{
    private sealed partial class CompilationContext
    {
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

            if (instruction.Opcode == "VReadfirstlaneB32")
            {
                if (instruction.Destinations.Count == 0 ||
                    instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister ||
                    instruction.Sources.Count == 0)
                {
                    error = "invalid read-first-lane operands";
                    return false;
                }

                var value = GetRawSource(instruction, 0);
                if (_subgroupInvocationIdInput != 0)
                {
                    if (_emulateWave64)
                    {
                        value = BroadcastFirstWave64Active(value);
                    }
                    else
                    {
                        // SPIR-V's BroadcastFirst uses the first host-active
                        // invocation. Guest EXEC is modeled as data, so obtain the
                        // guest-active mask explicitly and broadcast from its first
                        // set lane instead. This also updates the private SGPR copy
                        // for lanes that are currently disabled and may be restored
                        // by a later saveexec sequence.
                        var activeLanes = _module.AddInstruction(
                            SpirvOp.GroupNonUniformBallot,
                            _uvec4Type,
                            UInt(3),
                            Load(_boolType, _exec));
                        var activeLow = _module.AddInstruction(
                            SpirvOp.CompositeExtract,
                            _uintType,
                            activeLanes,
                            0);
                        var firstActiveLane = Ext(73, _uintType, activeLow);
                        value = _module.AddInstruction(
                            SpirvOp.GroupNonUniformBroadcast,
                            _uintType,
                            UInt(3),
                            value,
                            firstActiveLane);
                    }
                }

                StoreS(instruction.Destinations[0].Value, value);
                return true;
            }

            if (instruction.Opcode == "VReadlaneB32")
            {
                return TryEmitReadlane(instruction, out error);
            }

            if (instruction.Opcode is "VSwapB32" or "VSwaprelB32")
            {
                return TryEmitVectorSwap(instruction, out error);
            }

            if (instruction.Opcode is
                "VMovreldB32" or "VMovrelsB32" or
                "VMovrelsdB32" or "VMovrelsd2B32")
            {
                return TryEmitVectorRelativeMove(instruction, out error);
            }

            if (instruction.Opcode is
                "VLshlrevB64" or "VLshrrevB64" or "VAshrrevI64")
            {
                return TryEmitVector64Shift(instruction, out error);
            }

            if (instruction.Opcode is
                "VQsadPkU16U8" or "VMqsadPkU16U8" or "VMqsadU32U8")
            {
                return TryEmitPackedSad(instruction, out error);
            }

            if (!TryGetVectorDestination(instruction, out var destination))
            {
                error = "missing vector destination";
                return false;
            }

            uint result;
            switch (instruction.Opcode)
            {
                case "VMovB32":
                    result = GetRawSource(instruction, 0);
                    break;
                case "VWritelaneB32":
                {
                    // vdst[lane(src1)] = src0
                    // Per-lane: if current lane == src1, write src0, else keep old value.
                    var oldValue = LoadV(destination);
                    var src0 = GetRawSource(instruction, 0);
                    var laneSelect = GetRawSource(instruction, 1);
                    var currentLane = _subgroupInvocationIdInput != 0
                        ? BitwiseAnd(
                            Load(_uintType, _subgroupInvocationIdInput),
                            UInt(RdnaWaveLaneCount - 1))
                        : UInt(0);
                    var isTargetLane = _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        currentLane,
                        laneSelect);
                    result = _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        isTargetLane,
                        src0,
                        oldValue);
                    // Writelane writes to a specific lane regardless of exec mask.
                    StoreV(destination, result, guardWithExec: false);
                    return true;
                }
                case "VCndmaskB32":
                {
                    var condition = instruction.Sources.Count > 2
                        ? IsCurrentLaneSet(GetRawSource64(instruction, 2))
                        : Load(_boolType, _vcc);
                    result = _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        condition,
                        GetRawSource(instruction, 1),
                        GetRawSource(instruction, 0));
                    break;
                }
                case "VCvtU32F32":
                    result = _module.AddInstruction(
                        SpirvOp.ConvertFToU,
                        _uintType,
                        GetFloatSource(instruction, 0));
                    break;
                case "VCvtI32F32":
                case "VCvtRpiI32F32":
                case "VCvtFlrI32F32":
                {
                    var source = GetFloatSource(instruction, 0);
                    if (instruction.Opcode == "VCvtRpiI32F32")
                    {
                        source = Ext(9, _floatType, source);
                    }
                    else if (instruction.Opcode == "VCvtFlrI32F32")
                    {
                        source = Ext(8, _floatType, source);
                    }

                    result = Bitcast(
                        _uintType,
                        _module.AddInstruction(SpirvOp.ConvertFToS, _intType, source));
                    break;
                }
                case "VCvtI32F64":
                    result = EmitFloat64ToInt32(instruction, signed: true);
                    break;
                case "VCvtU32F64":
                    result = EmitFloat64ToInt32(instruction, signed: false);
                    break;
                case "VCvtF64I32":
                    return EmitFloat64FromInt32(instruction, destination, signed: true, out error);
                case "VCvtF64U32":
                    return EmitFloat64FromInt32(instruction, destination, signed: false, out error);
                case "VCvtF64F32":
                    return EmitFloat64FromF32(instruction, destination, out error);
                case "VCvtF32F64":
                    result = EmitFloat32FromF64(instruction);
                    break;
                case "VCvtF32I32":
                {
                    var signed = Bitcast(_intType, GetRawSource(instruction, 0));
                    result = Bitcast(
                        _uintType,
                        _module.AddInstruction(SpirvOp.ConvertSToF, _floatType, signed));
                    break;
                }
                case "VCvtF32U32":
                    result = Bitcast(
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.ConvertUToF,
                            _floatType,
                            GetRawSource(instruction, 0)));
                    break;
                case "VCvtF32Ubyte0":
                case "VCvtF32Ubyte1":
                case "VCvtF32Ubyte2":
                case "VCvtF32Ubyte3":
                {
                    var shift = (uint)(instruction.Opcode[^1] - '0') * 8;
                    var raw = ShiftRightLogical(GetRawSource(instruction, 0), UInt(shift));
                    raw = BitwiseAnd(raw, UInt(0xFF));
                    result = Bitcast(
                        _uintType,
                        _module.AddInstruction(SpirvOp.ConvertUToF, _floatType, raw));
                    break;
                }
                case "VCvtF16F32":
                {
                    var vector = _module.AddInstruction(
                        SpirvOp.CompositeConstruct,
                        _vec2Type,
                        GetFloatSource(instruction, 0),
                        Float(0));
                    result = BitwiseAnd(Ext(58, _uintType, vector), UInt(0xFFFF));
                    break;
                }
                case "VCvtF32F16":
                {
                    var unpacked = Ext(62, _vec2Type, GetRawSource(instruction, 0));
                    var value = _module.AddInstruction(
                        SpirvOp.CompositeExtract,
                        _floatType,
                        unpacked,
                        0);
                    result = Bitcast(_uintType, value);
                    break;
                }
                case "VCvtOffF32I4":
                    result = EmitCvtOffF32I4(instruction);
                    break;
                case "VCvtPkU8F32":
                {
                    var converted = _module.AddInstruction(
                        SpirvOp.ConvertFToU,
                        _uintType,
                        GetFloatSource(instruction, 0));
                    var offset = ShiftLeftLogical(
                        BitwiseAnd(GetRawSource(instruction, 1), UInt(3)),
                        UInt(3));
                    result = _module.AddInstruction(
                        SpirvOp.BitFieldInsert,
                        _uintType,
                        GetRawSource(instruction, 2),
                        converted,
                        offset,
                        UInt(8));
                    break;
                }
                case "VRcpF32":
                case "VRcpIflagF32":
                    result = EmitFloatResult(
                        instruction,
                        _module.AddInstruction(
                            SpirvOp.FDiv,
                            _floatType,
                            Float(1),
                            GetFloatSource(instruction, 0)));
                    break;
                case "VLogF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(30, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VLdexpF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(
                            53,
                            _floatType,
                            GetFloatSource(instruction, 0),
                            Bitcast(_intType, GetRawSource(instruction, 1))));
                    break;
                case "VExpF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(29, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VRsqF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(32, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VFractF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(10, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VTruncF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(3, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VCeilF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(9, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VRndneF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(2, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VFloorF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(8, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VFrexpExpI32F32":
                    result = EmitFrexpExponentF32(
                        instruction,
                        Bitcast(_uintType, GetFloatSource(instruction, 0)));
                    break;
                case "VFrexpMantF32":
                    result = EmitFloatResult(
                        instruction,
                        Bitcast(
                            _floatType,
                            EmitFrexpMantissaF32(
                                instruction,
                                Bitcast(_uintType, GetFloatSource(instruction, 0)))));
                    break;
                case "VFrexpExpI32F64":
                    result = EmitFrexpExponentF64(
                        instruction,
                        GetFloat64SourceBits(instruction, 0));
                    break;
                case "VFrexpMantF64":
                    return EmitFrexpMantissaF64(instruction, destination, out error);
                case "VTruncF64":
                    return EmitFloat64Round(instruction, destination, Float64RoundMode.Trunc, out error);
                case "VCeilF64":
                    return EmitFloat64Round(instruction, destination, Float64RoundMode.Ceil, out error);
                case "VRndneF64":
                    return EmitFloat64Round(instruction, destination, Float64RoundMode.NearestEven, out error);
                case "VFloorF64":
                    return EmitFloat64Round(instruction, destination, Float64RoundMode.Floor, out error);
                case "VFractF64":
                    return EmitFloat64Fract(instruction, destination, out error);
                case "VSqrtF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(31, _floatType, GetFloatSource(instruction, 0)));
                    break;
                case "VSinF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(
                            13,
                            _floatType,
                            _module.AddInstruction(
                                SpirvOp.FMul,
                                _floatType,
                                GetFloatSource(instruction, 0),
                                Float(MathF.Tau))));
                    break;
                case "VCosF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(
                            14,
                            _floatType,
                            _module.AddInstruction(
                                SpirvOp.FMul,
                                _floatType,
                                GetFloatSource(instruction, 0),
                                Float(MathF.Tau))));
                    break;
                case "VAddF32":
                    result = EmitFloatBinary(instruction, SpirvOp.FAdd);
                    break;
                case "VSubF32":
                    result = EmitFloatBinary(instruction, SpirvOp.FSub);
                    break;
                case "VSubrevF32":
                    result = EmitFloatBinary(instruction, SpirvOp.FSub, reverse: true);
                    break;
                case "VMulLegacyF32":
                    result = EmitLegacyFloatMultiply(instruction);
                    break;
                case "VMacLegacyF32":
                    result = EmitLegacyFloatMultiplyAccumulate(instruction, destination);
                    break;
                case "VMadLegacyF32":
                    result = EmitLegacyFloatMad(instruction);
                    break;
                case "VMullitF32":
                    result = EmitMullitF32(instruction);
                    break;
                case "VMulF32":
                    result = EmitFloatBinary(instruction, SpirvOp.FMul);
                    break;
                case "VMinF32":
                    result = EmitFloatExtBinary(instruction, 37);
                    break;
                case "VMaxF32":
                    result = EmitFloatExtBinary(instruction, 40);
                    break;
                case "VMadF32":
                case "VFmaF32":
                case "VMadMkF32":
                case "VMadAkF32":
                case "VFmaMkF32":
                case "VFmaAkF32":
                    result = EmitFloatResult(
                        instruction,
                        Ext(
                            50,
                            _floatType,
                            GetFloatSource(instruction, 0),
                            GetFloatSource(instruction, 1),
                            GetFloatSource(instruction, 2)));
                    break;
                case "VMacF32":
                case "VFmacF32":
                {
                    var addend = Bitcast(_floatType, LoadV(destination));
                    result = EmitFloatResult(
                        instruction,
                        Ext(
                            50,
                            _floatType,
                            GetFloatSource(instruction, 0),
                            GetFloatSource(instruction, 1),
                            addend));
                    break;
                }
                case "VMin3F32":
                    result = EmitFloatTernaryExt(instruction, 37);
                    break;
                case "VMax3F32":
                    result = EmitFloatTernaryExt(instruction, 40);
                    break;
                case "VFmaF16":
                case "VMin3F16":
                case "VMin3I16":
                case "VMin3U16":
                case "VMax3F16":
                case "VMax3I16":
                case "VMax3U16":
                case "VMed3F16":
                case "VMed3I16":
                case "VMed3U16":
                    if (!TryEmitVop3Half(instruction, destination, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VAddNcU16":
                case "VSubNcU16":
                case "VMulLoU16":
                case "VLshrrevB16":
                case "VAshrrevI16":
                case "VMaxU16":
                case "VMaxI16":
                case "VMinU16":
                case "VMinI16":
                case "VAddNcI16":
                case "VSubNcI16":
                case "VLshlrevB16":
                case "VMadU16":
                case "VMadI16":
                    if (!TryEmitVop3Integer16(instruction, destination, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VDivFixupF16":
                    if (!TryEmitDivFixupF16(instruction, destination, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VDivFmasF32":
                {
                    var fused = Ext(
                        50,
                        _floatType,
                        GetFloatSource(instruction, 0),
                        GetFloatSource(instruction, 1),
                        GetFloatSource(instruction, 2));
                    var scaled = _module.AddInstruction(
                        SpirvOp.FMul,
                        _floatType,
                        fused,
                        Float(4294967296f)); // 2^32, exactly representable in f32.
                    var selected = _module.AddInstruction(
                        SpirvOp.Select,
                        _floatType,
                        Load(_boolType, _vcc),
                        scaled,
                        fused);
                    result = EmitFloatResult(instruction, selected);
                    break;
                }
                case "VPackB32F16":
                case "VCvtPknormI16F16":
                case "VCvtPknormU16F16":
                    if (instruction.Control is not Gen5Vop3Control halfPackControl)
                    {
                        error = $"missing vop3 control for {instruction.Opcode}";
                        return false;
                    }

                    if (instruction.Opcode == "VPackB32F16")
                    {
                        result = BitwiseOr(
                            EmitVop3HalfBits(instruction, halfPackControl, 0),
                            ShiftLeftLogical(
                                EmitVop3HalfBits(instruction, halfPackControl, 1),
                                UInt(16)));
                    }
                    else
                    {
                        var vector = _module.AddInstruction(
                            SpirvOp.CompositeConstruct,
                            _vec2Type,
                            EmitVop3F16Operand(instruction, halfPackControl, 0),
                            EmitVop3F16Operand(instruction, halfPackControl, 1));
                        result = Ext(
                            instruction.Opcode == "VCvtPknormI16F16" ? 56u : 57u,
                            _uintType,
                            vector);
                    }

                    break;
                case "VAddF16":
                case "VSubF16":
                case "VSubrevF16":
                case "VMulF16":
                case "VFmacF16":
                case "VFmaMkF16":
                case "VFmaAkF16":
                case "VMaxF16":
                case "VMinF16":
                case "VLdexpF16":
                case "VRcpF16":
                case "VSqrtF16":
                case "VRsqF16":
                case "VLogF16":
                case "VExpF16":
                case "VFrexpMantF16":
                case "VFloorF16":
                case "VCeilF16":
                case "VTruncF16":
                case "VRndneF16":
                case "VFractF16":
                case "VSinF16":
                case "VCosF16":
                    if (!TryEmitScalarF16(instruction, destination, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VCvtF16U16":
                case "VCvtF16I16":
                case "VCvtU16F16":
                case "VCvtI16F16":
                    if (!TryEmitScalarF16Conversion(
                            instruction,
                            destination,
                            out result,
                            out error))
                    {
                        return false;
                    }

                    break;
                case "VFrexpExpI16F16":
                    result = MergeScalar16Result(
                        instruction,
                        destination,
                        EmitHalfFrexpExponentBits(
                            EmitScalarF16OperandBits(instruction, 0)));
                    break;
                case "VCvtNormI16F16":
                case "VCvtNormU16F16":
                {
                    var vector = _module.AddInstruction(
                        SpirvOp.CompositeConstruct,
                        _vec2Type,
                        EmitScalarF16Operand(instruction, 0),
                        Float(0));
                    var packed = Ext(
                        instruction.Opcode == "VCvtNormI16F16" ? 56u : 57u,
                        _uintType,
                        vector);
                    result = MergeScalar16Result(instruction, destination, packed);
                    break;
                }
                case "VSatPkU8I16":
                    result = EmitSatPkU8I16(instruction);
                    break;
                case "VPkFmacF16":
                    result = EmitPackedF16Accumulate(instruction, destination);
                    break;
                case "VAndB32":
                    result = EmitIntegerBinary(instruction, SpirvOp.BitwiseAnd);
                    break;
                case "VOrB32":
                    result = EmitIntegerBinary(instruction, SpirvOp.BitwiseOr);
                    break;
                case "VXorB32":
                    result = EmitIntegerBinary(instruction, SpirvOp.BitwiseXor);
                    break;
                case "VXnorB32":
                {
                    var xor = EmitIntegerBinary(instruction, SpirvOp.BitwiseXor);
                    result = _module.AddInstruction(SpirvOp.Not, _uintType, xor);
                    break;
                }
                case "VNotB32":
                    result = _module.AddInstruction(
                        SpirvOp.Not,
                        _uintType,
                        GetRawSource(instruction, 0));
                    break;
                case "VBfrevB32":
                    result = _module.AddInstruction(
                        SpirvOp.BitReverse,
                        _uintType,
                        GetRawSource(instruction, 0));
                    break;
                case "VFfblB32":
                    result = Bitcast(
                        _uintType,
                        Ext(
                            73,
                            _intType,
                            Bitcast(_intType, GetRawSource(instruction, 0))));
                    break;
                case "VFfbhU32":
                    result = EmitFindFirstBitHigh(instruction, signed: false);
                    break;
                case "VFfbhI32":
                    result = EmitFindFirstBitHigh(instruction, signed: true);
                    break;
                case "VAddI32":
                case "VAddU32":
                    result = EmitIntegerBinary(instruction, SpirvOp.IAdd);
                    break;
                case "VAddcU32":
                case "VAddCoCiU32":
                    result = EmitAddWithCarry(instruction);
                    break;
                case "VSubI32":
                case "VSubU32":
                    result = EmitIntegerBinary(instruction, SpirvOp.ISub);
                    break;
                case "VSubrevI32":
                case "VSubrevU32":
                case "VSubrevNcU32":
                    result = EmitIntegerBinary(instruction, SpirvOp.ISub, reverse: true);
                    break;
                case "VAddNcI32":
                case "VAddNcU32":
                    result = EmitIntegerBinary(instruction, SpirvOp.IAdd);
                    break;
                case "VSubNcI32":
                case "VSubNcU32":
                    result = EmitIntegerBinary(instruction, SpirvOp.ISub);
                    break;
                case "VSubbU32":
                    result = EmitSubtractWithBorrow(instruction, reverse: false);
                    break;
                case "VSubbrevU32":
                    result = EmitSubtractWithBorrow(instruction, reverse: true);
                    break;
                case "VMulLoU32":
                case "VMulLoI32":
                case "VMulU32U24":
                    result = EmitIntegerBinary(instruction, SpirvOp.IMul);
                    break;
                case "VMulI32I24":
                    result = EmitSigned24Product(instruction, high: false);
                    break;
                case "VMulHiI32I24":
                    result = EmitSigned24Product(instruction, high: true);
                    break;
                case "VMulHiU32":
                case "VMulHiU32U24":
                {
                    var left = GetRawSource(instruction, 0);
                    var right = GetRawSource(instruction, 1);
                    if (instruction.Opcode == "VMulHiU32U24")
                    {
                        left = BitwiseAnd(left, UInt(0x00FF_FFFF));
                        right = BitwiseAnd(right, UInt(0x00FF_FFFF));
                    }

                    var wideLeft = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        left);
                    var wideRight = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        right);
                    var product = _module.AddInstruction(
                        SpirvOp.IMul,
                        _ulongType,
                        wideLeft,
                        wideRight);
                    result = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _uintType,
                        ShiftRightLogical64(
                            product,
                            _module.Constant64(_ulongType, 32)));
                    break;
                }
                case "VMulHiI32":
                {
                    var wideLeft = _module.AddInstruction(
                        SpirvOp.SConvert,
                        _longType,
                        Bitcast(_intType, GetRawSource(instruction, 0)));
                    var wideRight = _module.AddInstruction(
                        SpirvOp.SConvert,
                        _longType,
                        Bitcast(_intType, GetRawSource(instruction, 1)));
                    var product = _module.AddInstruction(
                        SpirvOp.IMul,
                        _longType,
                        wideLeft,
                        wideRight);
                    result = Bitcast(
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.SConvert,
                            _intType,
                            _module.AddInstruction(
                                SpirvOp.ShiftRightArithmetic,
                                _longType,
                                product,
                                _module.Constant64(_longType, 32))));
                    break;
                }
                case "VBcntU32B32":
                    result = IAdd(
                        _module.AddInstruction(
                            SpirvOp.BitCount,
                            _uintType,
                            GetRawSource(instruction, 0)),
                        GetRawSource(instruction, 1));
                    break;
                case "VMbcntHiU32B32":
                {
                    var guestLane = GuestWaveLane();
                    var lane = BitwiseAnd(guestLane, UInt(31));
                    var partialMask = _module.AddInstruction(
                        SpirvOp.ISub,
                        _uintType,
                        ShiftLeftLogical(UInt(1), lane),
                        UInt(1));
                    var countedBits = _module.AddInstruction(
                        SpirvOp.BitCount,
                        _uintType,
                        BitwiseAnd(GetRawSource(instruction, 0), partialMask));
                    var isUpperHalf = _module.AddInstruction(
                        SpirvOp.UGreaterThanEqual,
                        _boolType,
                        guestLane,
                        UInt(32));
                    result = IAdd(
                        GetRawSource(instruction, 1),
                        _module.AddInstruction(
                            SpirvOp.Select,
                            _uintType,
                            isUpperHalf,
                            countedBits,
                            UInt(0)));
                    break;
                }
                case "VMbcntLoU32B32":
                {
                    var guestLane = GuestWaveLane();
                    var lane = BitwiseAnd(guestLane, UInt(31));
                    var partialMask = _module.AddInstruction(
                        SpirvOp.ISub,
                        _uintType,
                        ShiftLeftLogical(UInt(1), lane),
                        UInt(1));
                    var lowBitsMask = _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.UGreaterThanEqual,
                            _boolType,
                            guestLane,
                            UInt(32)),
                        UInt(uint.MaxValue),
                        partialMask);
                    result = IAdd(
                        GetRawSource(instruction, 1),
                        _module.AddInstruction(
                            SpirvOp.BitCount,
                            _uintType,
                            BitwiseAnd(GetRawSource(instruction, 0), lowBitsMask)));
                    break;
                }
                case "VBfmB32":
                {
                    var width = BitwiseAnd(GetRawSource(instruction, 0), UInt(31));
                    var lowMask = _module.AddInstruction(
                        SpirvOp.ISub,
                        _uintType,
                        ShiftLeftLogical(UInt(1), width),
                        UInt(1));
                    result = ShiftLeftLogical(lowMask, GetRawSource(instruction, 1));
                    break;
                }
                case "VMadU32U24":
                {
                    var left = BitwiseAnd(
                        GetRawSource(instruction, 0),
                        UInt(0x00FF_FFFF));
                    var right = BitwiseAnd(
                        GetRawSource(instruction, 1),
                        UInt(0x00FF_FFFF));
                    result = IAdd(
                        _module.AddInstruction(
                            SpirvOp.IMul,
                            _uintType,
                            left,
                            right),
                        GetRawSource(instruction, 2));
                    break;
                }
                case "VMadI32I24":
                    result = IAdd(
                        EmitSigned24Product(instruction, high: false),
                        GetRawSource(instruction, 2));
                    break;
                case "VMadU32U16":
                {
                    if (instruction.Control is not Gen5Vop3Control control)
                    {
                        error = "missing vop3 control for VMadU32U16";
                        return false;
                    }

                    var left = EmitVop3HalfBits(instruction, control, 0);
                    var right = EmitVop3HalfBits(instruction, control, 1);
                    result = IAdd(
                        _module.AddInstruction(
                            SpirvOp.IMul,
                            _uintType,
                            left,
                            right),
                        GetRawSource(instruction, 2));
                    break;
                }
                case "VLshrB32":
                    result = EmitIntegerBinary(instruction, SpirvOp.ShiftRightLogical);
                    break;
                case "VLshrrevB32":
                    result = EmitIntegerBinary(
                        instruction,
                        SpirvOp.ShiftRightLogical,
                        reverse: true);
                    break;
                case "VLshlB32":
                    result = EmitIntegerBinary(instruction, SpirvOp.ShiftLeftLogical);
                    break;
                case "VLshlrevB32":
                    result = EmitIntegerBinary(
                        instruction,
                        SpirvOp.ShiftLeftLogical,
                        reverse: true);
                    break;
                case "VAshrI32":
                case "VAshrrevI32":
                {
                    var reverse = instruction.Opcode == "VAshrrevI32";
                    var left = GetRawSource(instruction, reverse ? 1 : 0);
                    var right = GetRawSource(instruction, reverse ? 0 : 1);
                    result = ShiftRightArithmetic(left, right);
                    break;
                }
                case "VLshlAddU32":
                {
                    var shifted = ShiftLeftLogical(
                        GetRawSource(instruction, 0),
                        BitwiseAnd(GetRawSource(instruction, 1), UInt(31)));
                    result = IAdd(shifted, GetRawSource(instruction, 2));
                    break;
                }
                case "VLshlOrU32":
                case "VLshlOrB32":
                {
                    var shifted = ShiftLeftLogical(
                        GetRawSource(instruction, 0),
                        BitwiseAnd(GetRawSource(instruction, 1), UInt(31)));
                    result = BitwiseOr(
                        shifted,
                        GetRawSource(instruction, 2));
                    break;
                }
                case "VAndOrB32":
                    result = BitwiseOr(
                        BitwiseAnd(
                            GetRawSource(instruction, 0),
                            GetRawSource(instruction, 1)),
                        GetRawSource(instruction, 2));
                    break;
                case "VOr3U32":
                case "VOr3B32":
                    result = BitwiseOr(
                        BitwiseOr(
                            GetRawSource(instruction, 0),
                            GetRawSource(instruction, 1)),
                        GetRawSource(instruction, 2));
                    break;
                case "VPermlane16B32":
                    result = EmitPermlane16(instruction, exchangeRows: false);
                    break;
                case "VPermlanex16B32":
                    result = EmitPermlane16(instruction, exchangeRows: true);
                    break;
                case "VAddLshlU32":
                {
                    var added = IAdd(
                        GetRawSource(instruction, 0),
                        GetRawSource(instruction, 1));
                    result = ShiftLeftLogical(added, GetRawSource(instruction, 2));
                    break;
                }
                case "VAdd3U32":
                    result = IAdd(
                        IAdd(
                            GetRawSource(instruction, 0),
                            GetRawSource(instruction, 1)),
                        GetRawSource(instruction, 2));
                    break;
                case "VMinU32":
                    result = Ext(
                        38,
                        _uintType,
                        GetRawSource(instruction, 0),
                        GetRawSource(instruction, 1));
                    break;
                case "VMaxU32":
                    result = Ext(
                        41,
                        _uintType,
                        GetRawSource(instruction, 0),
                        GetRawSource(instruction, 1));
                    break;
                case "VMin3U32":
                    result = Ext(
                        38,
                        _uintType,
                        Ext(
                            38,
                            _uintType,
                            GetRawSource(instruction, 0),
                            GetRawSource(instruction, 1)),
                        GetRawSource(instruction, 2));
                    break;
                case "VMax3U32":
                    result = Ext(
                        41,
                        _uintType,
                        Ext(
                            41,
                            _uintType,
                            GetRawSource(instruction, 0),
                            GetRawSource(instruction, 1)),
                        GetRawSource(instruction, 2));
                    break;
                case "VMinI32":
                case "VMaxI32":
                {
                    var signedResult = Ext(
                        instruction.Opcode == "VMinI32" ? 39u : 42u,
                        _intType,
                        Bitcast(_intType, GetRawSource(instruction, 0)),
                        Bitcast(_intType, GetRawSource(instruction, 1)));
                    result = Bitcast(_uintType, signedResult);
                    break;
                }
                case "VMin3I32":
                case "VMax3I32":
                {
                    var operation = instruction.Opcode == "VMin3I32" ? 39u : 42u;
                    var left = Bitcast(
                        _intType,
                        GetRawSource(instruction, 0));
                    var middle = Bitcast(
                        _intType,
                        GetRawSource(instruction, 1));
                    var right = Bitcast(
                        _intType,
                        GetRawSource(instruction, 2));
                    result = Bitcast(
                        _uintType,
                        Ext(
                            operation,
                            _intType,
                            Ext(operation, _intType, left, middle),
                            right));
                    break;
                }
                case "VMed3U32":
                {
                    var left = GetRawSource(instruction, 0);
                    var middle = GetRawSource(instruction, 1);
                    var right = GetRawSource(instruction, 2);
                    var low = Ext(38, _uintType, left, middle);
                    var high = Ext(41, _uintType, left, middle);
                    result = Ext(
                        41,
                        _uintType,
                        low,
                        Ext(38, _uintType, high, right));
                    break;
                }
                case "VMed3I32":
                {
                    var left = Bitcast(_intType, GetRawSource(instruction, 0));
                    var middle = Bitcast(_intType, GetRawSource(instruction, 1));
                    var right = Bitcast(_intType, GetRawSource(instruction, 2));
                    var low = Ext(39, _intType, left, middle);
                    var high = Ext(42, _intType, left, middle);
                    result = Bitcast(
                        _uintType,
                        Ext(
                            42,
                            _intType,
                            low,
                            Ext(39, _intType, high, right)));
                    break;
                }
                case "VMed3F32":
                {
                    var left = GetFloatSource(instruction, 0);
                    var middle = GetFloatSource(instruction, 1);
                    var right = GetFloatSource(instruction, 2);
                    var low = Ext(37, _floatType, left, middle);
                    var high = Ext(40, _floatType, left, middle);
                    result = EmitFloatResult(
                        instruction,
                        Ext(
                            40,
                            _floatType,
                            low,
                            Ext(37, _floatType, high, right)));
                    break;
                }
                case "VCubeidF32":
                    result = EmitCubeCoordinate(instruction, CubeCoordinate.Id);
                    break;
                case "VCubescF32":
                    result = EmitCubeCoordinate(instruction, CubeCoordinate.Sc);
                    break;
                case "VCubetcF32":
                    result = EmitCubeCoordinate(instruction, CubeCoordinate.Tc);
                    break;
                case "VCubemaF32":
                    result = EmitCubeCoordinate(instruction, CubeCoordinate.Ma);
                    break;
                case "VAddCoU32":
                {
                    var left = GetRawSource(instruction, 0);
                    var right = GetRawSource(instruction, 1);
                    result = IAdd(left, right);
                    var carry = _module.AddInstruction(
                        SpirvOp.ULessThan,
                        _boolType,
                        result,
                        left);
                    StoreCarryOut(instruction, carry);
                    break;
                }
                case "VSubCoU32":
                case "VSubrevCoU32":
                {
                    var reverse = instruction.Opcode == "VSubrevCoU32";
                    var left = GetRawSource(instruction, reverse ? 1 : 0);
                    var right = GetRawSource(instruction, reverse ? 0 : 1);
                    result = _module.AddInstruction(SpirvOp.ISub, _uintType, left, right);
                    var borrow = _module.AddInstruction(
                        SpirvOp.ULessThan,
                        _boolType,
                        left,
                        right);
                    StoreCarryOut(instruction, borrow);
                    break;
                }
                case "VMadU64U32":
                {
                    // V_MAD_U64_U32 writes a 64-bit VGPR pair. The first two
                    // sources are 32-bit factors; the third is a 64-bit addend
                    // held in a VGPR or SGPR pair. Its SDST receives the carry
                    // mask for the unsigned 64-bit addition.
                    var wideLeft = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        GetRawSource(instruction, 0));
                    var wideRight = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        GetRawSource(instruction, 1));
                    var product = _module.AddInstruction(
                        SpirvOp.IMul,
                        _ulongType,
                        wideLeft,
                        wideRight);
                    var addend = GetRawSource64(instruction, 2);
                    var wideResult = _module.AddInstruction(
                        SpirvOp.IAdd,
                        _ulongType,
                        product,
                        addend);
                    var carry = _module.AddInstruction(
                        SpirvOp.ULessThan,
                        _boolType,
                        wideResult,
                        addend);
                    result = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _uintType,
                        wideResult);
                    var high = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _uintType,
                        ShiftRightLogical64(
                            wideResult,
                            _module.Constant64(_ulongType, 32)));
                    StoreV(destination + 1, high);
                    StoreCarryOut(instruction, carry);
                    break;
                }
                case "VBfeU32":
                {
                    var width = BitwiseAnd(GetRawSource(instruction, 2), UInt(31));
                    result = _module.AddInstruction(
                        SpirvOp.BitFieldUExtract,
                        _uintType,
                        GetRawSource(instruction, 0),
                        BitwiseAnd(GetRawSource(instruction, 1), UInt(31)),
                        width);
                    break;
                }
                case "VBfeI32":
                {
                    // Same extract as VBfeU32 but sign-extended from the top bit
                    // of the extracted field, so the result type must be signed
                    // and bitcast back for storage.
                    var width = BitwiseAnd(GetRawSource(instruction, 2), UInt(31));
                    result = Bitcast(
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.BitFieldSExtract,
                            _intType,
                            Bitcast(_intType, GetRawSource(instruction, 0)),
                            BitwiseAnd(GetRawSource(instruction, 1), UInt(31)),
                            width));
                    break;
                }
                case "VBfiB32":
                {
                    var mask = GetRawSource(instruction, 0);
                    var insert = GetRawSource(instruction, 1);
                    var source = GetRawSource(instruction, 2);
                    result = _module.AddInstruction(
                        SpirvOp.BitwiseOr,
                        _uintType,
                        BitwiseAnd(mask, insert),
                        BitwiseAnd(
                         _module.AddInstruction(SpirvOp.Not, _uintType, mask),
                             source));
                    break;
                }
                case "VLerpU8":
                    result = EmitLerpU8(instruction);
                    break;
                case "VXor3B32":
                    result = BitwiseXor(
                        BitwiseXor(
                            GetRawSource(instruction, 0),
                            GetRawSource(instruction, 1)),
                        GetRawSource(instruction, 2));
                    break;
                 case "VMadI32I16":
                {
                    if (instruction.Control is not Gen5Vop3Control control)
                    {
                        error = "missing vop3 control for VMadI32I16";
                        return false;
                    }

                    uint Signed16(int sourceIndex) => Bitcast(
                        _intType,
                        ShiftRightArithmetic(
                            ShiftLeftLogical(
                                EmitVop3HalfBits(instruction, control, sourceIndex),
                                UInt(16)),
                            UInt(16)));
                    var value = _module.AddInstruction(
                        SpirvOp.IAdd,
                        _intType,
                        _module.AddInstruction(
                            SpirvOp.IMul,
                            _intType,
                            Signed16(0),
                            Signed16(1)),
                        Bitcast(_intType, GetRawSource(instruction, 2)));
                    result = Bitcast(_uintType, value);
                    break;
                }
                case "VXadU32":
                {
                    result = IAdd(
                        BitwiseXor(
                            GetRawSource(instruction, 0),
                            GetRawSource(instruction, 1)),
                        GetRawSource(instruction, 2));
                    break;
                }
                case "VPermB32":
                {
                    var high = GetRawSource(instruction, 0);
                    var low = GetRawSource(instruction, 1);
                    var selectors = GetRawSource(instruction, 2);
                    result = UInt(0);
                    for (var byteIndex = 0; byteIndex < 4; byteIndex++)
                    {
                        var selector = BitwiseAnd(
                            ShiftRightLogical(selectors, UInt((uint)(byteIndex * 8))),
                            UInt(0xFF));
                        var value = EmitPermuteByte(high, low, selector);
                        result = BitwiseOr(
                            result,
                            ShiftLeftLogical(value, UInt((uint)(byteIndex * 8))));
                    }

                    break;
                }
                case "VAlignbitB32":
                case "VAlignbyteB32":
                {
                    var high = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        GetRawSource(instruction, 0));
                    var low = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        GetRawSource(instruction, 1));
                    var concatenated = BitwiseOr64(
                        ShiftLeftLogical64(
                            high,
                            _module.Constant64(_ulongType, 32)),
                        low);
                    var sourceCount = BitwiseAnd(
                        GetRawSource(instruction, 2),
                        UInt(31));
                    var shift = instruction.Opcode == "VAlignbyteB32"
                        ? _module.AddInstruction(
                            SpirvOp.IMul,
                            _uintType,
                            sourceCount,
                            UInt(8))
                        : sourceCount;
                    var shifted = ShiftRightLogical64(
                        concatenated,
                        _module.AddInstruction(
                            SpirvOp.UConvert,
                            _ulongType,
                            BitwiseAnd(shift, UInt(63))));
                    var narrowed = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _uintType,
                        shifted);
                    result = instruction.Opcode == "VAlignbyteB32"
                        ? _module.AddInstruction(
                            SpirvOp.Select,
                            _uintType,
                            _module.AddInstruction(
                                SpirvOp.UGreaterThanEqual,
                                _boolType,
                                sourceCount,
                                UInt(8)),
                            UInt(0),
                            narrowed)
                        : narrowed;
                    break;
                }
                case "VCvtPkrtzF16F32":
                {
                    var first = TruncateFloat32ForPack(GetFloatSource(instruction, 0));
                    var second = TruncateFloat32ForPack(GetFloatSource(instruction, 1));
                    var vector = _module.AddInstruction(
                        SpirvOp.CompositeConstruct,
                        _vec2Type,
                        first,
                        second);
                    result = Ext(58, _uintType, vector);
                    StorePackedHalf(
                        destination,
                        vector);
                    break;
                }
                case "VCvtPknormI16F32":
                case "VCvtPknormU16F32":
                {
                    var vector = _module.AddInstruction(
                        SpirvOp.CompositeConstruct,
                        _vec2Type,
                        GetFloatSource(instruction, 0),
                        GetFloatSource(instruction, 1));
                    // GLSL.std.450 PackSnorm2x16 / PackUnorm2x16 match the
                    // RDNA saturated normalized conversion and packing rules.
                    result = Ext(
                        instruction.Opcode == "VCvtPknormI16F32" ? 56u : 57u,
                        _uintType,
                        vector);
                    break;
                }
                case "VCvtPkU16U32":
                case "VCvtPkI16I32":
                    result = BitwiseOr(
                        BitwiseAnd(GetRawSource(instruction, 0), UInt(0xFFFF)),
                        ShiftLeftLogical(
                            BitwiseAnd(GetRawSource(instruction, 1), UInt(0xFFFF)),
                            UInt(16)));
                    break;
                case "VDot2cF32F16":
                {
                    var source0 = GetRawSource(instruction, 0);
                    var source1 = GetRawSource(instruction, 1);
                    var source0Low = Bitcast(
                        _floatType,
                        EmitHalfToFloat(BitwiseAnd(source0, UInt(0xFFFF))));
                    var source0High = Bitcast(
                        _floatType,
                        EmitHalfToFloat(ShiftRightLogical(source0, UInt(16))));
                    var source1Low = Bitcast(
                        _floatType,
                        EmitHalfToFloat(BitwiseAnd(source1, UInt(0xFFFF))));
                    var source1High = Bitcast(
                        _floatType,
                        EmitHalfToFloat(ShiftRightLogical(source1, UInt(16))));
                    var accumulated = _module.AddInstruction(
                        SpirvOp.FAdd,
                        _floatType,
                        Bitcast(_floatType, LoadV(destination)),
                        _module.AddInstruction(
                            SpirvOp.FMul,
                            _floatType,
                            source0Low,
                            source1Low));
                    accumulated = _module.AddInstruction(
                        SpirvOp.FAdd,
                        _floatType,
                        accumulated,
                        _module.AddInstruction(
                            SpirvOp.FMul,
                            _floatType,
                            source0High,
                            source1High));
                    result = EmitFloatResult(instruction, accumulated);
                    break;
                }
                case "VSadU8":
                case "VSadHiU8":
                case "VSadU16":
                case "VSadU32":
                    result = EmitUnsignedSad(instruction);
                    break;
                case "VMsadU8":
                    result = EmitMaskedUnsignedSadU8(instruction);
                    break;

                case "VPkMadI16":
                case "VPkMulLoU16":
                case "VPkAddI16":
                case "VPkSubI16":
                case "VPkLshlrevB16":
                case "VPkLshrrevB16":
                case "VPkAshrrevI16":
                case "VPkMaxI16":
                case "VPkMinI16":
                case "VPkMadU16":
                case "VPkAddU16":
                case "VPkSubU16":
                case "VPkMaxU16":
                case "VPkMinU16":
                    if (!TryEmitPackedInteger16(instruction, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VDot2I32I16":
                case "VDot2U32U16":
                case "VDot4I32I8":
                case "VDot4U32U8":
                case "VDot8I32I4":
                case "VDot8U32U4":
                    if (!TryEmitPackedIntegerDot(instruction, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VDot2F32F16":
                    if (!TryEmitPackedFloatDot(instruction, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VPkAddF16":
                case "VPkMulF16":
                case "VPkMinF16":
                case "VPkMaxF16":
                case "VPkFmaF16":
                    if (!TryEmitPackedF16(instruction, out result, out error))
                    {
                        return false;
                    }

                    break;
                case "VFmaMixF32":
                case "VFmaMixloF16":
                case "VFmaMixhiF16":
                    if (!TryEmitFmaMix(instruction, destination, out result, out error))
                    {
                        return false;
                    }

                    break;
                default:
                    error = $"unsupported vector opcode {instruction.Opcode}";
                    return false;
            }

            if (instruction.Control is Gen5DppControl dpp)
            {
                result = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    IsDppWriteEnabled(dpp),
                    result,
                    LoadV(destination));
            }

            if (instruction.Control is Gen5SdwaControl destinationControl &&
                destinationControl.ScalarDestination is null)
            {
                result = ApplySdwaDestination(
                    destinationControl,
                    result,
                    LoadV(destination));
            }

            StoreV(destination, result);
            return true;
        }

        private bool TryEmitVectorRelativeMove(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (!TryGetVectorDestination(instruction, out var destination) ||
                instruction.Sources.Count != 1)
            {
                error = $"invalid {instruction.Opcode} operands";
                return false;
            }

            // RDNA2 6.6: M0 is an unsigned VGPR index.  The RELS forms require
            // SRC0 to name a VGPR base; RELD retains ordinary VOP1 source
            // selection and only indexes the destination.
            var relativeSource = instruction.Opcode is
                "VMovrelsB32" or "VMovrelsdB32" or "VMovrelsd2B32";
            if (relativeSource &&
                instruction.Sources[0].Kind != Gen5OperandKind.VectorRegister)
            {
                error = $"{instruction.Opcode} expects a VGPR source base";
                return false;
            }

            var m0 = LoadS(124);
            var splitOffsets = instruction.Opcode == "VMovrelsd2B32";
            var sourceOffset = splitOffsets
                ? BitwiseAnd(m0, UInt(0x3FF))
                : m0;
            var destinationOffset = splitOffsets
                ? BitwiseAnd(ShiftRightLogical(m0, UInt(16)), UInt(0x3FF))
                : m0;
            var value = relativeSource
                ? LoadVRelative(instruction.Sources[0].Value, sourceOffset)
                : GetRawSource(instruction, 0);

            if (instruction.Opcode is
                "VMovreldB32" or "VMovrelsdB32" or "VMovrelsd2B32")
            {
                StoreVRelative(destination, destinationOffset, value);
            }
            else
            {
                StoreV(destination, value);
            }

            return true;
        }

        private bool TryEmitVector64Shift(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (!TryGetVectorDestination(instruction, out var destination) ||
                instruction.Sources.Count < 2)
            {
                error = $"invalid {instruction.Opcode} operands";
                return false;
            }

            // GFX10's REV forms take the shift count in SRC0 and the 64-bit
            // value in SRC1.  The count is masked to six bits before the
            // 64-bit shift; the arithmetic form preserves the sign bit of the
            // assembled pair while the logical forms operate on raw bits.
            var shift32 = BitwiseAnd(GetRawSource(instruction, 0), UInt(63));
            var shiftUnsigned = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                shift32);
            var value = GetRawSource64(instruction, 1);
            var shifted = instruction.Opcode switch
            {
                "VLshlrevB64" => ShiftLeftLogical64(value, shiftUnsigned),
                "VLshrrevB64" => ShiftRightLogical64(value, shiftUnsigned),
                _ => _module.AddInstruction(
                    SpirvOp.ShiftRightArithmetic,
                    _longType,
                    Bitcast(_longType, value),
                    Bitcast(_longType, shiftUnsigned)),
            };

            var low = _module.AddInstruction(SpirvOp.UConvert, _uintType, shifted);
            var high = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(
                    shifted,
                    _module.Constant64(_ulongType, 32)));
            StoreV(destination + 1, high);
            StoreV(destination, low);
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
                var sourceValue = LoadV(sourceBase);
                var destinationValue = LoadV(destinationBase);
                StoreV(destinationBase, sourceValue);
                StoreV(sourceBase, destinationValue);
                return true;
            }

            var m0 = LoadS(124);
            var sourceOffset = BitwiseAnd(m0, UInt(0x3FF));
            var destinationOffset = BitwiseAnd(
                ShiftRightLogical(m0, UInt(16)),
                UInt(0x3FF));
            var relativeSourceValue = LoadVRelative(sourceBase, sourceOffset);
            var relativeDestinationValue = LoadVRelative(
                destinationBase,
                destinationOffset);
            StoreVRelative(destinationBase, destinationOffset, relativeSourceValue);
            StoreVRelative(sourceBase, sourceOffset, relativeDestinationValue);
            return true;
        }

        // V_SAD_* computes unsigned per-component absolute differences and then
        // accumulates them into src2. Integer VOP3 clamp means saturating the
        // unsigned result rather than applying the floating-point [0, 1] clamp.
        private uint EmitUnsignedSad(Gen5ShaderInstruction instruction)
        {
            var source0 = GetRawSource(instruction, 0);
            var source1 = GetRawSource(instruction, 1);
            var source2 = GetRawSource(instruction, 2);
            var clamp = instruction.Control is Gen5Vop3Control { Clamp: true };

            uint SumComponents(int componentBits, int componentCount)
            {
                var mask = UInt(componentBits == 8 ? 0xFFu : 0xFFFFu);
                var sum = UInt(0);
                for (var component = 0; component < componentCount; component++)
                {
                    var shift = UInt((uint)(component * componentBits));
                    var left = BitwiseAnd(ShiftRightLogical(source0, shift), mask);
                    var right = BitwiseAnd(ShiftRightLogical(source1, shift), mask);
                    sum = IAdd(sum, EmitUnsignedAbsDiff(left, right));
                }

                return sum;
            }

            return instruction.Opcode switch
            {
                "VSadU8" => EmitUnsignedAdd(source2, SumComponents(8, 4), clamp),
                "VSadHiU8" => EmitUnsignedAdd(
                    source2,
                    ShiftLeftLogical(SumComponents(8, 4), UInt(16)),
                    clamp),
                "VSadU16" => EmitUnsignedAdd(source2, SumComponents(16, 2), clamp),
                "VSadU32" => EmitUnsignedAdd(
                    source2,
                    EmitUnsignedAbsDiff(source0, source1),
                    clamp),
                _ => source2,
            };
        }

        private uint EmitUnsignedAbsDiff(uint left, uint right)
        {
            var leftIsGreater = _module.AddInstruction(
                SpirvOp.UGreaterThan,
                _boolType,
                left,
                right);
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                leftIsGreater,
                _module.AddInstruction(SpirvOp.ISub, _uintType, left, right),
                _module.AddInstruction(SpirvOp.ISub, _uintType, right, left));
        }

        // V_MSAD_U8 is the masked byte form of V_SAD_U8.  RDNA2 copies S2
        // first, then adds each unsigned byte absolute difference only when
        // the corresponding byte of S1 is non-zero.
        private uint EmitMaskedUnsignedSadU8(Gen5ShaderInstruction instruction)
        {
            var source0 = GetRawSource(instruction, 0);
            var source1 = GetRawSource(instruction, 1);
            var result = GetRawSource(instruction, 2);

            return IAdd(result, EmitUnsignedSadBytes(source0, source1, masked: true));
        }

        private uint EmitUnsignedSadBytes(uint source0, uint source1, bool masked)
        {
            var result = UInt(0);
            for (var component = 0; component < 4; component++)
            {
                var shift = UInt((uint)(component * 8));
                var left = BitwiseAnd(
                    ShiftRightLogical(source0, shift),
                    UInt(0xFF));
                var right = BitwiseAnd(
                    ShiftRightLogical(source1, shift),
                    UInt(0xFF));
                var difference = EmitUnsignedAbsDiff(left, right);
                if (masked)
                {
                    var enabled = _module.AddInstruction(
                        SpirvOp.INotEqual,
                        _boolType,
                        right,
                        UInt(0));
                    difference = _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        enabled,
                        difference,
                        UInt(0));
                }

                result = IAdd(result, difference);
            }

            return result;
        }

        private bool TryEmitPackedSad(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (!TryGetVectorDestination(instruction, out var destination) ||
                instruction.Sources.Count < 3)
            {
                error = $"invalid {instruction.Opcode} operands";
                return false;
            }

            var source0 = GetRawSource64(instruction, 0);
            var source1 = GetRawSource(instruction, 1);
            var masked = instruction.Opcode is "VMqsadPkU16U8" or "VMqsadU32U8";

            uint Source2Dword(uint index) => instruction.Sources[2].Kind switch
            {
                Gen5OperandKind.VectorRegister => LoadV(instruction.Sources[2].Value + index),
                Gen5OperandKind.ScalarRegister => LoadS(instruction.Sources[2].Value + index),
                _ when index == 0 => GetRawSource(instruction, 2),
                _ => UInt(0),
            };

            uint Source0Chunk(uint index) => _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(
                    source0,
                    _module.Constant64(_ulongType, index * 8)));

            if (instruction.Opcode is "VQsadPkU16U8" or "VMqsadPkU16U8")
            {
                var source2 = GetRawSource64(instruction, 2);
                var low = UInt(0);
                var high = UInt(0);
                for (var component = 0; component < 4; component++)
                {
                    var sad = EmitUnsignedSadBytes(
                        Source0Chunk((uint)component),
                        source1,
                        masked);
                    var accumulator = BitwiseAnd(
                        _module.AddInstruction(
                            SpirvOp.UConvert,
                            _uintType,
                            ShiftRightLogical64(
                                source2,
                                _module.Constant64(
                                    _ulongType,
                                    (uint)(component * 16)))),
                        UInt(0xFFFF));
                    var packed = BitwiseAnd(
                        IAdd(accumulator, sad),
                        UInt(0xFFFF));
                    var shifted = ShiftLeftLogical(
                        packed,
                        UInt((uint)((component & 1) * 16)));
                    if (component < 2)
                    {
                        low = BitwiseOr(low, shifted);
                    }
                    else
                    {
                        high = BitwiseOr(
                            high,
                            ShiftLeftLogical(
                                packed,
                                UInt((uint)((component - 2) * 16))));
                    }
                }

                StoreV(destination + 1, high);
                StoreV(destination, low);
                return true;
            }

            // V_MQSAD_U32_U8 writes four independent dwords.  S0 is a
            // 64-bit pair; S2 supplies four dword accumulators.
            for (var component = 0; component < 4; component++)
            {
                var value = IAdd(
                    Source2Dword((uint)component),
                    EmitUnsignedSadBytes(
                        Source0Chunk((uint)component),
                        source1,
                        masked: true));
                StoreV(destination + (uint)component, value);
            }

            return true;
        }

        private uint EmitUnsignedAdd(uint left, uint right, bool saturate)
        {
            var sum = IAdd(left, right);
            if (!saturate)
            {
                return sum;
            }

            var overflow = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                sum,
                left);
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                overflow,
                UInt(uint.MaxValue),
                sum);
        }

        private uint EmitFindFirstBitHigh(
            Gen5ShaderInstruction instruction,
            bool signed)
        {
            var source = GetRawSource(instruction, 0);
            var minusOne = Bitcast(_intType, UInt(uint.MaxValue));
            var thirtyOne = Bitcast(_intType, UInt(31));
            var bitIndex = Ext(
                signed ? 74u : 75u,
                _intType,
                signed ? Bitcast(_intType, source) : source);
            var notFound = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                bitIndex,
                minusOne);
            var leadingCount = _module.AddInstruction(
                SpirvOp.ISub,
                _intType,
                thirtyOne,
                bitIndex);
            return Bitcast(
                _uintType,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _intType,
                    notFound,
                    minusOne,
                    leadingCount));
        }

        private bool TryEmitScalarF16(
            Gen5ShaderInstruction instruction,
            uint destination,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Sources.Count < 1)
            {
                error = $"invalid f16 operands for {instruction.Opcode}";
                return false;
            }

            var source0 = EmitScalarF16Operand(instruction, 0);
            var source1 = instruction.Sources.Count > 1
                ? EmitScalarF16Operand(instruction, 1)
                : 0u;
            uint valueBits;
            switch (instruction.Opcode)
            {
                case "VAddF16":
                    valueBits = Bitcast(
                        _uintType,
                        _module.AddInstruction(SpirvOp.FAdd, _floatType, source0, source1));
                    break;
                case "VSubF16":
                case "VSubrevF16":
                    valueBits = Bitcast(
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.FSub,
                            _floatType,
                            instruction.Opcode == "VSubrevF16" ? source1 : source0,
                            instruction.Opcode == "VSubrevF16" ? source0 : source1));
                    break;
                case "VMulF16":
                    valueBits = Bitcast(
                        _uintType,
                        _module.AddInstruction(SpirvOp.FMul, _floatType, source0, source1));
                    break;
                case "VFmacF16":
                {
                    var addend = EmitHalfBitsAsFloat(
                        SelectScalarF16DestinationHalf(instruction, destination));
                    valueBits = EmitPackedF16FusedMultiplyAdd(source0, source1, addend);
                    break;
                }
                case "VFmaMkF16":
                case "VFmaAkF16":
                    if (instruction.Sources.Count < 3)
                    {
                        error = $"missing literal/addend for {instruction.Opcode}";
                        return false;
                    }

                    valueBits = EmitPackedF16FusedMultiplyAdd(
                        source0,
                        source1,
                        EmitScalarF16Operand(instruction, 2));
                    break;
                case "VMaxF16":
                    valueBits = Bitcast(
                        _uintType,
                        EmitPackedF16MinMax(source0, source1, isMax: true));
                    break;
                case "VMinF16":
                    valueBits = Bitcast(
                        _uintType,
                        EmitPackedF16MinMax(source0, source1, isMax: false));
                    break;
                case "VLdexpF16":
                {
                    var exponentBits = EmitScalarF16SourceBits(instruction, 1);
                    var exponent = Bitcast(
                        _intType,
                        ShiftRightArithmetic(
                            ShiftLeftLogical(exponentBits, UInt(16)),
                            UInt(16)));
                    valueBits = Bitcast(
                        _uintType,
                        Ext(53, _floatType, source0, exponent));
                    break;
                }
                case "VRcpF16":
                    valueBits = Bitcast(
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.FDiv,
                            _floatType,
                            Float(1),
                            source0));
                    break;
                case "VSqrtF16":
                    valueBits = Bitcast(_uintType, Ext(31, _floatType, source0));
                    break;
                case "VRsqF16":
                    valueBits = Bitcast(_uintType, Ext(32, _floatType, source0));
                    break;
                case "VLogF16":
                    valueBits = Bitcast(_uintType, Ext(30, _floatType, source0));
                    break;
                case "VExpF16":
                    valueBits = Bitcast(_uintType, Ext(29, _floatType, source0));
                    break;
                case "VFrexpMantF16":
                    valueBits = Bitcast(
                        _uintType,
                        EmitHalfBitsAsFloat(
                            EmitHalfFrexpMantissaBits(
                                EmitScalarF16OperandBits(instruction, 0))));
                    break;
                case "VFloorF16":
                    valueBits = Bitcast(_uintType, Ext(8, _floatType, source0));
                    break;
                case "VCeilF16":
                    valueBits = Bitcast(_uintType, Ext(9, _floatType, source0));
                    break;
                case "VTruncF16":
                    valueBits = Bitcast(_uintType, Ext(3, _floatType, source0));
                    break;
                case "VRndneF16":
                    valueBits = Bitcast(_uintType, Ext(2, _floatType, source0));
                    break;
                case "VFractF16":
                    valueBits = Bitcast(_uintType, Ext(10, _floatType, source0));
                    break;
                case "VSinF16":
                case "VCosF16":
                {
                    var radians = _module.AddInstruction(
                        SpirvOp.FMul,
                        _floatType,
                        source0,
                        Float(2 * MathF.PI));
                    valueBits = Bitcast(
                        _uintType,
                        Ext(instruction.Opcode == "VSinF16" ? 13u : 14u, _floatType, radians));
                    break;
                }
                default:
                    error = $"unsupported scalar f16 opcode {instruction.Opcode}";
                    return false;
            }

            result = FinishScalarF16Result(instruction, destination, valueBits);
            return true;
        }

        private uint FinishScalarF16Result(
            Gen5ShaderInstruction instruction,
            uint destination,
            uint valueBits)
        {
            var control = instruction.Control switch
            {
                Gen5Vop3Control vop3 =>
                    (OutputModifier: vop3.OutputModifier, Clamp: vop3.Clamp),
                Gen5SdwaControl sdwa =>
                    (OutputModifier: sdwa.OutputModifier, Clamp: sdwa.Clamp),
                _ => (OutputModifier: 0u, Clamp: false),
            };
            var value = Bitcast(_floatType, valueBits);
            value = control.OutputModifier switch
            {
                1 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(2)),
                2 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(4)),
                3 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(0.5f)),
                _ => value,
            };
            valueBits = Bitcast(_uintType, value);
            if (control.Clamp)
            {
                valueBits = EmitClampToUnitInterval(valueBits);
            }

            var half = EmitFloatToHalf(valueBits);
            return MergeScalar16Result(instruction, destination, half);
        }

        private uint MergeScalar16Result(
            Gen5ShaderInstruction instruction,
            uint destination,
            uint half)
        {
            half = BitwiseAnd(half, UInt(0xFFFF));
            if (instruction.Control is Gen5Vop3Control vop3Control)
            {
                var existing = LoadV(destination);
                return (vop3Control.OperandSelect & 8) == 0
                    ? BitwiseOr(BitwiseAnd(existing, UInt(0xFFFF_0000)), half)
                    : BitwiseOr(
                        BitwiseAnd(existing, UInt(0x0000_FFFF)),
                        ShiftLeftLogical(half, UInt(16)));
            }
            else
            {
                // Native 16-bit VOP1/VOP2 operations define the high half as zero.
                return half;
            }
        }

        private bool TryEmitScalarF16Conversion(
            Gen5ShaderInstruction instruction,
            uint destination,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Sources.Count != 1)
            {
                error = $"invalid f16 conversion operands for {instruction.Opcode}";
                return false;
            }

            if (instruction.Opcode is "VCvtF16U16" or "VCvtF16I16")
            {
                var sourceBits = EmitScalarF16SourceBits(instruction, 0);
                uint value;
                if (instruction.Opcode == "VCvtF16I16")
                {
                    var signed = Bitcast(
                        _intType,
                        ShiftRightArithmetic(
                            ShiftLeftLogical(sourceBits, UInt(16)),
                            UInt(16)));
                    value = _module.AddInstruction(
                        SpirvOp.ConvertSToF,
                        _floatType,
                        signed);
                }
                else
                {
                    value = _module.AddInstruction(
                        SpirvOp.ConvertUToF,
                        _floatType,
                        sourceBits);
                }

                result = FinishScalarF16Result(
                    instruction,
                    destination,
                    Bitcast(_uintType, value));
                return true;
            }

            var source = EmitScalarF16Operand(instruction, 0);
            var isNan = _module.AddInstruction(SpirvOp.IsNan, _boolType, source);
            source = _module.AddInstruction(
                SpirvOp.Select,
                _floatType,
                isNan,
                Float(0),
                source);
            var signedResult = instruction.Opcode == "VCvtI16F16";
            source = Ext(
                43,
                _floatType,
                source,
                Float(signedResult ? -32768 : 0),
                Float(signedResult ? 32767 : 65535));
            uint integer;
            if (signedResult)
            {
                integer = Bitcast(
                    _uintType,
                    _module.AddInstruction(
                        SpirvOp.ConvertFToS,
                        _intType,
                        source));
            }
            else
            {
                integer = _module.AddInstruction(
                    SpirvOp.ConvertFToU,
                    _uintType,
                    source);
            }

            result = MergeScalar16Result(instruction, destination, integer);
            return true;
        }

        private uint EmitHalfFrexpMantissaBits(uint half)
        {
            var sign = BitwiseAnd(half, UInt(0x8000));
            var exponent = BitwiseAnd(half, UInt(0x7C00));
            var fraction = BitwiseAnd(half, UInt(0x03FF));
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0x7C00));
            var hasZeroExponent = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0));
            var hasZeroFraction = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                UInt(0));
            var mostSignificantBit = Bitcast(
                _uintType,
                Ext(75, _intType, fraction));
            var shift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(10),
                mostSignificantBit);
            var normalizedFraction = BitwiseAnd(
                ShiftLeftLogical(fraction, shift),
                UInt(0x03FF));
            var normal = BitwiseOr(
                BitwiseOr(sign, UInt(0x3800)),
                fraction);
            var subnormal = BitwiseOr(
                BitwiseOr(sign, UInt(0x3800)),
                normalizedFraction);
            var zeroOrSubnormal = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroFraction,
                half,
                subnormal);
            var finite = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroExponent,
                zeroOrSubnormal,
                normal);
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isSpecial,
                half,
                finite);
        }

        private uint EmitHalfFrexpExponentBits(uint half)
        {
            var exponent = BitwiseAnd(
                ShiftRightLogical(half, UInt(10)),
                UInt(0x1F));
            var fraction = BitwiseAnd(half, UInt(0x03FF));
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0x1F));
            var hasZeroExponent = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0));
            var hasZeroFraction = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                UInt(0));
            var normalExponent = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                exponent,
                UInt(14));
            var subnormalExponent = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                Bitcast(_uintType, Ext(75, _intType, fraction)),
                UInt(23));
            var finiteExponent = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroExponent,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    hasZeroFraction,
                    UInt(0),
                    subnormalExponent),
                normalExponent);
            return BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    isSpecial,
                    UInt(0),
                    finiteExponent),
                UInt(0xFFFF));
        }

        private uint EmitFrexpExponentF32(
            Gen5ShaderInstruction instruction,
            uint bits)
        {
            var exponent = BitwiseAnd(ShiftRightLogical(bits, UInt(23)), UInt(0xFF));
            var fraction = BitwiseAnd(bits, UInt(0x007F_FFFF));
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0xFF));
            var hasZeroFraction = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                UInt(0));
            var hasZeroExponent = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0));
            var msb = Bitcast(
                _uintType,
                Ext(75, _intType, fraction));
            var normalExponent = Bitcast(
                _uintType,
                _module.AddInstruction(
                    SpirvOp.ISub,
                    _intType,
                    Bitcast(_intType, exponent),
                    Bitcast(_intType, UInt(126))));
            var subnormalExponent = Bitcast(
                _uintType,
                _module.AddInstruction(
                    SpirvOp.ISub,
                    _intType,
                    Bitcast(_intType, msb),
                    Bitcast(_intType, UInt(148))));
            var finiteExponent = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroExponent,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    hasZeroFraction,
                    UInt(0),
                    subnormalExponent),
                normalExponent);
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isSpecial,
                UInt(0),
                finiteExponent);
        }

        private uint EmitFrexpMantissaF32(
            Gen5ShaderInstruction instruction,
            uint bits)
        {
            var sign = BitwiseAnd(bits, UInt(0x8000_0000));
            var exponent = BitwiseAnd(bits, UInt(0x7F80_0000));
            var fraction = BitwiseAnd(bits, UInt(0x007F_FFFF));
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0x7F80_0000));
            var hasZeroExponent = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0));
            var hasZeroFraction = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                UInt(0));
            var msb = Bitcast(
                _uintType,
                Ext(75, _intType, fraction));
            var safeMsb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroFraction,
                UInt(0),
                msb);
            var shift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(23),
                safeMsb);
            var normalizedFraction = BitwiseAnd(
                ShiftLeftLogical(fraction, shift),
                UInt(0x007F_FFFF));
            var normal = BitwiseOr(BitwiseOr(sign, UInt(0x3F00_0000)), fraction);
            var subnormal = BitwiseOr(
                BitwiseOr(sign, UInt(0x3F00_0000)),
                normalizedFraction);
            var zeroOrSubnormal = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroFraction,
                bits,
                subnormal);
            var finite = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroExponent,
                zeroOrSubnormal,
                normal);
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isSpecial,
                bits,
                finite);
        }

        private uint EmitFrexpExponentF64(
            Gen5ShaderInstruction instruction,
            uint bits)
        {
            var exponent = BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                    bits,
                        _module.Constant64(_ulongType, 52))),
                UInt(0x7FF));
            var fraction = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0x7FF));
            var hasZeroExponent = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0));
            var hasZeroFraction = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                _module.Constant64(_ulongType, 0));
            var fractionLow = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                fraction);
            var fractionHigh = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(
                    fraction,
                    _module.Constant64(_ulongType, 32)));
            var highMsb = Bitcast(_uintType, Ext(75, _intType, fractionHigh));
            var lowMsb = Bitcast(_uintType, Ext(75, _intType, fractionLow));
            var msb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                IsNotZero(fractionHigh),
                IAdd(highMsb, UInt(32)),
                lowMsb);
            var normalExponent = Bitcast(
                _uintType,
                _module.AddInstruction(
                    SpirvOp.ISub,
                    _intType,
                    Bitcast(_intType, exponent),
                    Bitcast(_intType, UInt(1022))));
            var subnormalExponent = Bitcast(
                _uintType,
                _module.AddInstruction(
                    SpirvOp.ISub,
                    _intType,
                    Bitcast(_intType, msb),
                    Bitcast(_intType, UInt(1073))));
            var finiteExponent = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroExponent,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    hasZeroFraction,
                    UInt(0),
                    subnormalExponent),
                normalExponent);
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isSpecial,
                UInt(0),
                finiteExponent);
        }

        private bool EmitFrexpMantissaF64(
            Gen5ShaderInstruction instruction,
            uint destination,
            out string error)
        {
            error = string.Empty;
            var bits = GetFloat64SourceBits(instruction, 0);
            var sign = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x8000_0000_0000_0000UL));
            var exponent = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x7FF0_0000_0000_0000UL));
            var fraction = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                _module.Constant64(_ulongType, 0x7FF0_0000_0000_0000UL));
            var hasZeroExponent = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                _module.Constant64(_ulongType, 0));
            var hasZeroFraction = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                _module.Constant64(_ulongType, 0));
            var fractionLow = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                fraction);
            var fractionHigh = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(
                    fraction,
                    _module.Constant64(_ulongType, 32)));
            var highMsb = Bitcast(_uintType, Ext(75, _intType, fractionHigh));
            var lowMsb = Bitcast(_uintType, Ext(75, _intType, fractionLow));
            var msb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                IsNotZero(fractionHigh),
                IAdd(highMsb, UInt(32)),
                lowMsb);
            var safeMsb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasZeroFraction,
                UInt(0),
                msb);
            var shift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(52),
                safeMsb);
            var normalizedFraction = BitwiseAnd64(
                ShiftLeftLogical64(fraction, _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    shift)),
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var normal = _module.AddInstruction(
                SpirvOp.BitwiseOr,
                _ulongType,
                BitwiseOr64(sign, _module.Constant64(_ulongType, 0x3FE0_0000_0000_0000UL)),
                fraction);
            var subnormal = _module.AddInstruction(
                SpirvOp.BitwiseOr,
                _ulongType,
                BitwiseOr64(sign, _module.Constant64(_ulongType, 0x3FE0_0000_0000_0000UL)),
                normalizedFraction);
            var zeroOrSubnormal = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                hasZeroFraction,
                bits,
                subnormal);
            var finite = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                hasZeroExponent,
                zeroOrSubnormal,
                normal);
            var result = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                isSpecial,
                bits,
                finite);
            StoreV(
                destination,
                _module.AddInstruction(SpirvOp.UConvert, _uintType, result));
            StoreV(
                destination + 1,
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        result,
                        _module.Constant64(_ulongType, 32))));
            return true;
        }

        private uint EmitFloat64ToInt32(
            Gen5ShaderInstruction instruction,
            bool signed)
        {
            // Keep this conversion entirely in the IEEE-754 bit domain.  Vulkan
            // implementations are allowed to expose no shaderFloat64 support,
            // while the RDNA2 conversion still has defined truncation and
            // saturation behaviour for finite values, infinities and NaNs.
            var bits = GetFloat64SourceBits(instruction, 0);
            var sign = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x8000_0000_0000_0000UL));
            var exponent = BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        bits,
                        _module.Constant64(_ulongType, 52))),
                UInt(0x7FF));
            var fraction = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var isNan = _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    exponent,
                    UInt(0x7FF)),
                IsNotZero64(fraction));
            var isNegative = IsNotZero64(sign);
            var significand = BitwiseOr64(
                fraction,
                _module.Constant64(_ulongType, 0x0010_0000_0000_0000UL));

            // The integer part is significand * 2^(exponent-1075).  Clamp
            // dynamic shift counts before they reach SPIR-V (which masks them
            // to 6 bits); values outside the representable range are selected
            // to the saturating result below anyway.
            var hasLeftShift = _module.AddInstruction(
                SpirvOp.UGreaterThanEqual,
                _boolType,
                exponent,
                UInt(1075));
            var leftRaw = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                exponent,
                UInt(1075));
            var leftClamped = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    leftRaw,
                    UInt(63)),
                UInt(63),
                leftRaw);
            var leftShift = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasLeftShift,
                leftClamped,
                UInt(0));
            var hasRightShift = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                exponent,
                UInt(1075));
            var rightRaw = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(1075),
                exponent);
            var rightClamped = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    rightRaw,
                    UInt(63)),
                UInt(63),
                rightRaw);
            var rightShift = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasRightShift,
                rightClamped,
                UInt(0));
            var leftShift64 = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                leftShift);
            var rightShift64 = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                rightShift);
            var leftMagnitude = ShiftLeftLogical64(significand, leftShift64);
            var rightMagnitude = ShiftRightLogical64(significand, rightShift64);
            var magnitude64 = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                hasLeftShift,
                leftMagnitude,
                rightMagnitude);
            var magnitude = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                magnitude64);
            var inRange = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                exponent,
                UInt(1054));

            uint finite;
            if (signed)
            {
                var signedMagnitude = _module.AddInstruction(
                    SpirvOp.ISub,
                    _uintType,
                    UInt(0),
                    magnitude);
                var truncated = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    isNegative,
                    signedMagnitude,
                    magnitude);
                var saturated = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    isNegative,
                    UInt(0x8000_0000),
                    UInt(0x7FFF_FFFF));
                finite = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    inRange,
                    truncated,
                    saturated);
            }
            else
            {
                var saturated = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    isNegative,
                    UInt(0),
                    UInt(uint.MaxValue));
                finite = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    inRange,
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        isNegative,
                        UInt(0),
                        magnitude),
                    saturated);
            }

            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isNan,
                UInt(0),
                finite);
        }

        private bool EmitFloat64FromInt32(
            Gen5ShaderInstruction instruction,
            uint destination,
            bool signed,
            out string error)
        {
            error = string.Empty;
            var source = GetRawSource(instruction, 0);
            var negative = signed
                ? IsNotZero(BitwiseAnd(source, UInt(0x8000_0000)))
                : _module.ConstantBool(false);
            var magnitude = signed
                ? _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    negative,
                    _module.AddInstruction(SpirvOp.ISub, _uintType, UInt(0), source),
                    source)
                : source;
            var isZero = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                magnitude,
                UInt(0));
            var msb = Bitcast(
                _uintType,
                Ext(75, _intType, Bitcast(_intType, magnitude)));
            var safeMsb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isZero,
                UInt(0),
                msb);
            var shift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(52),
                safeMsb);
            var fraction = BitwiseAnd64(
                ShiftLeftLogical64(
                    _module.AddInstruction(SpirvOp.UConvert, _ulongType, magnitude),
                    _module.AddInstruction(SpirvOp.UConvert, _ulongType, shift)),
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var exponent = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                IAdd(UInt(1023), safeMsb));
            var signBits = signed
                ? _module.AddInstruction(
                    SpirvOp.Select,
                    _ulongType,
                    negative,
                    _module.Constant64(_ulongType, 0x8000_0000_0000_0000UL),
                    _module.Constant64(_ulongType, 0))
                : _module.Constant64(_ulongType, 0);
            var result = BitwiseOr64(
                signBits,
                BitwiseOr64(
                    ShiftLeftLogical64(
                        exponent,
                        _module.Constant64(_ulongType, 52)),
                    fraction));
            result = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                isZero,
                _module.Constant64(_ulongType, 0),
                result);
            StoreV(
                destination,
                _module.AddInstruction(SpirvOp.UConvert, _uintType, result));
            StoreV(
                destination + 1,
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        result,
                        _module.Constant64(_ulongType, 32))));
            return true;
        }

        private bool EmitFloat64FromF32(
            Gen5ShaderInstruction instruction,
            uint destination,
            out string error)
        {
            error = string.Empty;
            var bits = Bitcast(_uintType, GetFloatSource(instruction, 0));
            var sign = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                ShiftRightLogical(bits, UInt(31)));
            sign = ShiftLeftLogical64(sign, _module.Constant64(_ulongType, 63));
            var exponent = BitwiseAnd(
                ShiftRightLogical(bits, UInt(23)),
                UInt(0xFF));
            var fraction = BitwiseAnd(bits, UInt(0x007F_FFFF));
            var isZero = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                UInt(0));
            var msb = Bitcast(
                _uintType,
                Ext(75, _intType, Bitcast(_intType, fraction)));
            var safeMsb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isZero,
                UInt(0),
                msb);
            var subnormalShift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(52),
                safeMsb);
            var subnormalFraction = BitwiseAnd64(
                ShiftLeftLogical64(
                    _module.AddInstruction(SpirvOp.UConvert, _ulongType, fraction),
                    _module.AddInstruction(SpirvOp.UConvert, _ulongType, subnormalShift)),
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var normalFraction = ShiftLeftLogical64(
                _module.AddInstruction(SpirvOp.UConvert, _ulongType, fraction),
                _module.Constant64(_ulongType, 29));
            var normalExponent = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                IAdd(exponent, UInt(896)));
            var subnormalExponent = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                IAdd(safeMsb, UInt(874)));
            var normal = BitwiseOr64(
                sign,
                BitwiseOr64(
                    ShiftLeftLogical64(
                        normalExponent,
                        _module.Constant64(_ulongType, 52)),
                    normalFraction));
            var subnormal = BitwiseOr64(
                sign,
                BitwiseOr64(
                    ShiftLeftLogical64(
                        subnormalExponent,
                        _module.Constant64(_ulongType, 52)),
                    subnormalFraction));
            var finite = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    exponent,
                    UInt(0)),
                _module.AddInstruction(
                    SpirvOp.Select,
                    _ulongType,
                    isZero,
                    sign,
                    subnormal),
                normal);
            var special = BitwiseOr64(
                sign,
                BitwiseOr64(
                    _module.Constant64(_ulongType, 0x7FF0_0000_0000_0000UL),
                    ShiftLeftLogical64(
                        _module.AddInstruction(SpirvOp.UConvert, _ulongType, fraction),
                        _module.Constant64(_ulongType, 29))));
            var result = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    exponent,
                    UInt(0xFF)),
                special,
                finite);
            StoreV(
                destination,
                _module.AddInstruction(SpirvOp.UConvert, _uintType, result));
            StoreV(
                destination + 1,
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        result,
                        _module.Constant64(_ulongType, 32))));
            return true;
        }

        private uint EmitFloat32FromF64(Gen5ShaderInstruction instruction)
        {
            var bits = GetFloat64SourceBits(instruction, 0);
            var sign = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(
                    bits,
                    _module.Constant64(_ulongType, 32)));
            sign = BitwiseAnd(sign, UInt(0x8000_0000));
            var magnitudeBits = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x7FFF_FFFF_FFFF_FFFFUL));
            var exponent = BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        magnitudeBits,
                        _module.Constant64(_ulongType, 52))),
                UInt(0x7FF));
            var fraction = BitwiseAnd64(
                magnitudeBits,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var significand = BitwiseOr64(
                fraction,
                _module.Constant64(_ulongType, 0x0010_0000_0000_0000UL));
            var normalShift64 = _module.Constant64(_ulongType, 29);
            var normalRetained = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(significand, normalShift64));
            var normalRemainder = BitwiseAnd64(
                significand,
                _module.Constant64(_ulongType, 0x1FFF_FFFFUL));
            var normalHalf = _module.Constant64(_ulongType, 0x1000_0000UL);
            var normalRound = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    normalRemainder,
                    normalHalf),
                _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        normalRemainder,
                        normalHalf),
                    IsNotZero(BitwiseAnd(normalRetained, UInt(1)))));
            var normalRounded = IAdd(
                normalRetained,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    normalRound,
                    UInt(1),
                    UInt(0)));
            var normalCarry = _module.AddInstruction(
                SpirvOp.UGreaterThanEqual,
                _boolType,
                normalRounded,
                UInt(0x0100_0000));
            var normalExponent = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                exponent,
                UInt(896));
            normalExponent = IAdd(
                normalExponent,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    normalCarry,
                    UInt(1),
                    UInt(0)));
            var normalFraction = BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    normalCarry,
                    ShiftRightLogical(normalRounded, UInt(1)),
                    normalRounded),
                UInt(0x007F_FFFF));
            var normalBits = BitwiseOr(
                sign,
                BitwiseOr(
                    ShiftLeftLogical(
                        BitwiseAnd(normalExponent, UInt(0xFF)),
                        UInt(23)),
                    normalFraction));
            normalBits = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThanEqual,
                    _boolType,
                    exponent,
                    UInt(1151)),
                BitwiseOr(sign, UInt(0x7F80_0000)),
                normalBits);

            var subnormalShift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(926),
                exponent);
            var subnormalShiftClamped = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    subnormalShift,
                    UInt(63)),
                UInt(63),
                subnormalShift);
            var subnormalShift64 = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                subnormalShiftClamped);
            var subnormalRetained = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(significand, subnormalShift64));
            var subnormalMask = _module.AddInstruction(
                SpirvOp.ISub,
                _ulongType,
                ShiftLeftLogical64(
                    _module.Constant64(_ulongType, 1),
                    subnormalShift64),
                _module.Constant64(_ulongType, 1));
            var subnormalRemainder = BitwiseAnd64(significand, subnormalMask);
            var subnormalHalf = ShiftRightLogical64(
                subnormalMask,
                _module.Constant64(_ulongType, 1));
            var subnormalRound = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    subnormalRemainder,
                    subnormalHalf),
                _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        subnormalRemainder,
                        subnormalHalf),
                    IsNotZero(BitwiseAnd(subnormalRetained, UInt(1)))));
            var subnormalRounded = IAdd(
                subnormalRetained,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    subnormalRound,
                    UInt(1),
                    UInt(0)));
            var subnormalCarry = _module.AddInstruction(
                SpirvOp.UGreaterThanEqual,
                _boolType,
                subnormalRounded,
                UInt(0x0080_0000));
            var subnormalBits = BitwiseOr(
                sign,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    subnormalCarry,
                    UInt(0x0080_0000),
                    subnormalRounded));
            var finiteBits = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.ULessThan,
                    _boolType,
                    exponent,
                    UInt(897)),
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        exponent,
                        UInt(0)),
                    sign,
                    subnormalBits),
                normalBits);
            var specialFraction = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(fraction, normalShift64));
            var isInfinity = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                fraction,
                _module.Constant64(_ulongType, 0));
            specialFraction = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                isInfinity,
                UInt(0),
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        specialFraction,
                        UInt(0)),
                    UInt(1),
                    BitwiseAnd(specialFraction, UInt(0x007F_FFFF))));
            var specialBits = BitwiseOr(
                sign,
                BitwiseOr(
                    UInt(0x7F80_0000),
                    specialFraction));
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    exponent,
                    UInt(0x7FF)),
                specialBits,
                finiteBits);
        }

        private bool EmitFloat64Fract(
            Gen5ShaderInstruction instruction,
            uint destination,
            out string error)
        {
            error = string.Empty;
            var bits = GetFloat64SourceBits(instruction, 0);
            var sign = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x8000_0000_0000_0000UL));
            var magnitude = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x7FFF_FFFF_FFFF_FFFFUL));
            var exponent = BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        magnitude,
                        _module.Constant64(_ulongType, 52))),
                UInt(0x7FF));
            var fraction = BitwiseAnd64(
                magnitude,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var significand = BitwiseOr64(
                fraction,
                _module.Constant64(_ulongType, 0x0010_0000_0000_0000UL));
            var belowOne = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                exponent,
                UInt(1023));
            var belowInteger = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                exponent,
                UInt(1075));
            var shiftRaw = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(1075),
                exponent);
            var shiftClamped = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    shiftRaw,
                    UInt(63)),
                UInt(63),
                shiftRaw);
            var shift64 = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                shiftClamped);
            var remainderMask = BitwiseAnd64(
                _module.AddInstruction(
                    SpirvOp.ISub,
                    _ulongType,
                    ShiftLeftLogical64(
                        _module.Constant64(_ulongType, 1),
                        shift64),
                    _module.Constant64(_ulongType, 1)),
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var remainder = BitwiseAnd64(significand, remainderMask);
            var remainderMsb = FindMsb64(remainder);
            var remainderSafeMsb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                IsNotZero64(remainder),
                remainderMsb,
                UInt(0));
            var remainderFraction = BitwiseAnd64(
                ShiftLeftLogical64(
                    remainder,
                    _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.ISub,
                            _uintType,
                            UInt(52),
                            remainderSafeMsb))),
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var remainderExponent = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                _module.AddInstruction(
                    SpirvOp.IAdd,
                    _uintType,
                    _module.AddInstruction(
                        SpirvOp.ISub,
                        _uintType,
                        exponent,
                        UInt(52)),
                    remainderSafeMsb));
            var normalizedRemainder = BitwiseOr64(
                ShiftLeftLogical64(
                    remainderExponent,
                    _module.Constant64(_ulongType, 52)),
                remainderFraction);
            var positiveFraction = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                belowOne,
                magnitude,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _ulongType,
                    belowInteger,
                    normalizedRemainder,
                    _module.Constant64(_ulongType, 0)));

            // For negative inputs DX-style fract is 1 - frac(abs(x)).  Work at
            // a fixed 2^-53 scale, which is sufficient to round the result to a
            // double without ever constructing a hardware double value.
            var yExponent = BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        positiveFraction,
                        _module.Constant64(_ulongType, 52))),
                UInt(0x7FF));
            var yFraction = BitwiseAnd64(
                positiveFraction,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var ySignificand = BitwiseOr64(
                yFraction,
                _module.Constant64(_ulongType, 0x0010_0000_0000_0000UL));
            var ySmall = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                yExponent,
                UInt(970));
            var yShift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(1022),
                yExponent);
            var yShift64 = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                yShift);
            var yMask = _module.AddInstruction(
                SpirvOp.ISub,
                _ulongType,
                ShiftLeftLogical64(
                    _module.Constant64(_ulongType, 1),
                    yShift64),
                _module.Constant64(_ulongType, 1));
            var yRetained = ShiftRightLogical64(ySignificand, yShift64);
            var yRemainder = BitwiseAnd64(ySignificand, yMask);
            var yHalf = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    yShift,
                    UInt(0)),
                _module.Constant64(_ulongType, 0),
                ShiftRightLogical64(
                    yMask,
                    _module.Constant64(_ulongType, 1)));
            var yRound = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    yRemainder,
                    yHalf),
                _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        yRemainder,
                        yHalf),
                    IsNotZero64(
                        BitwiseAnd64(
                            yRetained,
                            _module.Constant64(_ulongType, 1)))));
            var yUnits = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                ySmall,
                _module.Constant64(_ulongType, 0),
                _module.AddInstruction(
                    SpirvOp.IAdd,
                    _ulongType,
                    yRetained,
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _ulongType,
                        yRound,
                        _module.Constant64(_ulongType, 1),
                        _module.Constant64(_ulongType, 0))));
            var halfTie = _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    yExponent,
                    UInt(970)),
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    yFraction,
                    _module.Constant64(_ulongType, 0)));
            var difference = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                halfTie,
                _module.Constant64(_ulongType, 0x0020_0000_0000_0000UL),
                _module.AddInstruction(
                    SpirvOp.ISub,
                    _ulongType,
                    _module.Constant64(_ulongType, 0x0020_0000_0000_0000UL),
                    yUnits));
            var differenceMsb = FindMsb64(difference);
            var differenceSafeMsb = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    difference,
                    _module.Constant64(_ulongType, 0x0020_0000_0000_0000UL)),
                UInt(52),
                differenceMsb);
            var oneMinus = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    difference,
                    _module.Constant64(_ulongType, 0x0020_0000_0000_0000UL)),
                _module.Constant64(_ulongType, 0x3FF0_0000_0000_0000UL),
                BitwiseOr64(
                    ShiftLeftLogical64(
                        _module.AddInstruction(
                            SpirvOp.UConvert,
                            _ulongType,
                            _module.AddInstruction(
                                SpirvOp.IAdd,
                                _uintType,
                                UInt(970),
                                differenceSafeMsb)),
                        _module.Constant64(_ulongType, 52)),
                    BitwiseAnd64(
                        ShiftLeftLogical64(
                            difference,
                            _module.AddInstruction(
                                SpirvOp.UConvert,
                                _ulongType,
                                _module.AddInstruction(
                                    SpirvOp.ISub,
                                    _uintType,
                                    UInt(52),
                                    differenceSafeMsb))),
                        _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL))));
            var negativeFraction = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                IsNotZero64(positiveFraction),
                oneMinus,
                _module.Constant64(_ulongType, 0));
            var finite = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                IsNotZero64(sign),
                negativeFraction,
                positiveFraction);
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0x7FF));
            var special = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    fraction,
                    _module.Constant64(_ulongType, 0)),
                BitwiseOr64(
                    sign,
                    _module.Constant64(_ulongType, 0x7FF8_0000_0000_0000UL)),
                bits);
            var result = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                isSpecial,
                special,
                finite);
            StoreV(
                destination,
                _module.AddInstruction(SpirvOp.UConvert, _uintType, result));
            StoreV(
                destination + 1,
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        result,
                        _module.Constant64(_ulongType, 32))));
            return true;
        }

        private enum Float64RoundMode
        {
            Trunc,
            Ceil,
            NearestEven,
            Floor,
        }

        private bool EmitFloat64Round(
            Gen5ShaderInstruction instruction,
            uint destination,
            Float64RoundMode mode,
            out string error)
        {
            error = string.Empty;
            var bits = GetFloat64SourceBits(instruction, 0);
            var signMask = _module.Constant64(_ulongType, 0x8000_0000_0000_0000UL);
            var sign = BitwiseAnd64(bits, signMask);
            var magnitudeBits = BitwiseAnd64(
                bits,
                _module.Constant64(_ulongType, 0x7FFF_FFFF_FFFF_FFFFUL));
            var exponent = BitwiseAnd(
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        magnitudeBits,
                        _module.Constant64(_ulongType, 52))),
                UInt(0x7FF));
            var fraction = BitwiseAnd64(
                magnitudeBits,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var significand = BitwiseOr64(
                fraction,
                _module.Constant64(_ulongType, 0x0010_0000_0000_0000UL));
            var isSpecial = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                exponent,
                UInt(0x7FF));
            var isSubnormal = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                exponent,
                UInt(1023));
            var hasNormalFraction = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                exponent,
                UInt(1075));
            var normalShift = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                UInt(1075),
                exponent);
            var normalShift64 = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                normalShift);
            var normalMask = BitwiseAnd64(
                _module.AddInstruction(
                    SpirvOp.ISub,
                    _ulongType,
                    ShiftLeftLogical64(
                        _module.Constant64(_ulongType, 1),
                        normalShift64),
                    _module.Constant64(_ulongType, 1)),
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL));
            var truncMask = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                isSubnormal,
                _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL),
                _module.AddInstruction(
                    SpirvOp.Select,
                    _ulongType,
                    hasNormalFraction,
                    normalMask,
                    _module.Constant64(_ulongType, 0)));
            var truncatedMagnitude = BitwiseAnd64(
                magnitudeBits,
                _module.AddInstruction(
                    SpirvOp.Not,
                    _ulongType,
                    truncMask));
            var hasFraction = IsNotZero64(
                BitwiseAnd64(magnitudeBits, truncMask));

            uint increment;
            switch (mode)
            {
                case Float64RoundMode.Trunc:
                    increment = _module.ConstantBool(false);
                    break;
                case Float64RoundMode.Ceil:
                    increment = _module.AddInstruction(
                        SpirvOp.LogicalAnd,
                        _boolType,
                        _module.AddInstruction(
                            SpirvOp.LogicalNot,
                            _boolType,
                            IsNotZero64(sign)),
                        hasFraction);
                    break;
                case Float64RoundMode.Floor:
                    increment = _module.AddInstruction(
                        SpirvOp.LogicalAnd,
                        _boolType,
                        IsNotZero64(sign),
                        hasFraction);
                    break;
                default:
                {
                    var isAtLeastHalf = _module.AddInstruction(
                        SpirvOp.UGreaterThanEqual,
                        _boolType,
                        exponent,
                        UInt(1022));
                    var isRoundable = _module.AddInstruction(
                        SpirvOp.ULessThan,
                        _boolType,
                        exponent,
                        UInt(1075));
                    var halfShift = _module.AddInstruction(
                        SpirvOp.ISub,
                        _uintType,
                        UInt(1075),
                        exponent);
                    var halfShift64 = _module.AddInstruction(
                        SpirvOp.UConvert,
                        _ulongType,
                        halfShift);
                    var halfMask = _module.AddInstruction(
                        SpirvOp.ISub,
                        _ulongType,
                        ShiftLeftLogical64(
                            _module.Constant64(_ulongType, 1),
                            halfShift64),
                        _module.Constant64(_ulongType, 1));
                    var remainder = BitwiseAnd64(significand, halfMask);
                    var half = ShiftRightLogical64(
                        halfMask,
                        _module.Constant64(_ulongType, 1));
                    var greaterHalf = _module.AddInstruction(
                        SpirvOp.UGreaterThan,
                        _boolType,
                        remainder,
                        half);
                    var equalHalf = _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        remainder,
                        half);
                    var odd = _module.AddInstruction(
                        SpirvOp.ULessThan,
                        _boolType,
                        exponent,
                        UInt(1075));
                    var oddBit = IsNotZero64(
                        BitwiseAnd64(
                            ShiftRightLogical64(significand, halfShift64),
                            _module.Constant64(_ulongType, 1)));
                    increment = _module.AddInstruction(
                        SpirvOp.LogicalAnd,
                        _boolType,
                        isAtLeastHalf,
                        _module.AddInstruction(
                            SpirvOp.LogicalOr,
                            _boolType,
                            greaterHalf,
                            _module.AddInstruction(
                                SpirvOp.LogicalAnd,
                                _boolType,
                                equalHalf,
                                _module.AddInstruction(
                                    SpirvOp.LogicalAnd,
                                    _boolType,
                                    odd,
                                    oddBit))));
                    increment = _module.AddInstruction(
                        SpirvOp.LogicalAnd,
                        _boolType,
                        isRoundable,
                        increment);
                    break;
                }
            }

            var unit = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                isSubnormal,
                _module.Constant64(_ulongType, 0x3FF0_0000_0000_0000UL),
                _module.AddInstruction(
                    SpirvOp.Select,
                    _ulongType,
                    hasNormalFraction,
                    ShiftLeftLogical64(
                        _module.Constant64(_ulongType, 1),
                        normalShift64),
                    _module.Constant64(_ulongType, 0)));
            var roundedMagnitude = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                increment,
                _module.AddInstruction(
                    SpirvOp.IAdd,
                    _ulongType,
                    truncatedMagnitude,
                    unit),
                truncatedMagnitude);
            var result = BitwiseOr64(sign, roundedMagnitude);
            result = _module.AddInstruction(
                SpirvOp.Select,
                _ulongType,
                isSpecial,
                bits,
                result);
            StoreV(
                destination,
                _module.AddInstruction(SpirvOp.UConvert, _uintType, result));
            StoreV(
                destination + 1,
                _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    ShiftRightLogical64(
                        result,
                        _module.Constant64(_ulongType, 32))));
            return true;
        }

        private uint EmitSatPkU8I16(Gen5ShaderInstruction instruction)
        {
            var source = GetRawSource(
                instruction,
                0,
                applySdwaIntegerModifiers: false);
            uint ClampSignedHalf(uint bits)
            {
                var signed = Bitcast(
                    _intType,
                    ShiftRightArithmetic(
                        ShiftLeftLogical(bits, UInt(16)),
                        UInt(16)));
                return Bitcast(
                    _uintType,
                    Ext(
                        45,
                        _intType,
                        signed,
                        Bitcast(_intType, UInt(0)),
                        Bitcast(_intType, UInt(255))));
            }

            var low = ClampSignedHalf(BitwiseAnd(source, UInt(0xFFFF)));
            var high = ClampSignedHalf(ShiftRightLogical(source, UInt(16)));
            return BitwiseOr(
                BitwiseAnd(low, UInt(0xFF)),
                ShiftLeftLogical(BitwiseAnd(high, UInt(0xFF)), UInt(8)));
        }

        private uint EmitScalarF16Operand(Gen5ShaderInstruction instruction, int sourceIndex)
        {
            return EmitHalfBitsAsFloat(
                EmitScalarF16OperandBits(instruction, sourceIndex));
        }

        private uint EmitScalarF16OperandBits(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            var half = EmitScalarF16SourceBits(instruction, sourceIndex);
            uint absoluteMask = 0;
            uint negateMask = 0;
            switch (instruction.Control)
            {
                case Gen5Vop3Control control:
                    absoluteMask = control.AbsoluteMask;
                    negateMask = control.NegateMask;
                    break;
                case Gen5SdwaControl control:
                    absoluteMask = control.AbsoluteMask;
                    negateMask = control.NegateMask;
                    break;
                case Gen5DppControl control:
                    absoluteMask = control.AbsoluteMask;
                    negateMask = control.NegateMask;
                    break;
            }

            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                half = BitwiseAnd(half, UInt(0x7FFF));
            }

            if ((negateMask & (1u << sourceIndex)) != 0)
            {
                half = BitwiseXor(half, UInt(0x8000));
            }

            return half;
        }

        private uint EmitScalarF16SourceBits(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            var raw = GetRawSource(
                instruction,
                sourceIndex,
                applySdwaIntegerModifiers: false);
            if (instruction.Sources[sourceIndex] is
                { Kind: Gen5OperandKind.EncodedConstant, Value: >= 240 and <= 248 } constant &&
                Gen5InlineConstants.TryDecode(constant.Value, out var floatBits))
            {
                raw = UInt(BitConverter.HalfToUInt16Bits(
                    (Half)BitConverter.UInt32BitsToSingle(floatBits)));
            }

            if (instruction.Control is Gen5Vop3Control control &&
                (control.OperandSelect & (1u << sourceIndex)) != 0)
            {
                raw = ShiftRightLogical(raw, UInt(16));
            }

            return BitwiseAnd(raw, UInt(0xFFFF));
        }

        private uint SelectScalarF16DestinationHalf(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            var raw = LoadV(destination);
            return instruction.Control is Gen5Vop3Control { OperandSelect: var operandSelect } &&
                   (operandSelect & 8) != 0
                ? BitwiseAnd(ShiftRightLogical(raw, UInt(16)), UInt(0xFFFF))
                : BitwiseAnd(raw, UInt(0xFFFF));
        }

        private uint EmitHalfBitsAsFloat(uint half) =>
            Bitcast(_floatType, EmitHalfToFloat(half));

        private uint EmitPackedF16Accumulate(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            var source0 = GetRawSource(instruction, 0);
            var source1 = GetRawSource(instruction, 1);
            var existing = LoadV(destination);
            uint Lane(uint source, bool high) => EmitHalfBitsAsFloat(
                BitwiseAnd(high ? ShiftRightLogical(source, UInt(16)) : source, UInt(0xFFFF)));
            var low = EmitFloatToHalf(EmitPackedF16FusedMultiplyAdd(
                Lane(source0, high: false),
                Lane(source1, high: false),
                Lane(existing, high: false)));
            var high = EmitFloatToHalf(EmitPackedF16FusedMultiplyAdd(
                Lane(source0, high: true),
                Lane(source1, high: true),
                Lane(existing, high: true)));
            return BitwiseOr(low, ShiftLeftLogical(high, UInt(16)));
        }

        // Scalar f16/i16/u16 VOP3 arithmetic. OPSEL[0:2] picks the half of each
        // source and OPSEL[3] picks the destination half; the other destination
        // half is architecturally preserved. Keeping the operation in the existing
        // integer f16 conversion path avoids requiring Float16 storage/arithmetic
        // support from the host device.
        private bool TryEmitVop3Half(
            Gen5ShaderInstruction instruction,
            uint destination,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3Control control ||
                instruction.Sources.Count < 3)
            {
                error = $"invalid half-precision VOP3 operands for {instruction.Opcode}";
                return false;
            }

            uint halfResult;
            if (instruction.Opcode == "VFmaF16")
            {
                var fused = EmitPackedF16FusedMultiplyAdd(
                    EmitVop3F16Operand(instruction, control, 0),
                    EmitVop3F16Operand(instruction, control, 1),
                    EmitVop3F16Operand(instruction, control, 2));
                halfResult = EmitVop3F16Result(fused, control);
            }
            else if (instruction.Opcode.EndsWith("F16", StringComparison.Ordinal))
            {
                var source0 = EmitVop3F16Operand(instruction, control, 0);
                var source1 = EmitVop3F16Operand(instruction, control, 1);
                var source2 = EmitVop3F16Operand(instruction, control, 2);
                uint value;
                if (instruction.Opcode.StartsWith("VMin3", StringComparison.Ordinal))
                {
                    value = EmitPackedF16MinMax(
                        EmitPackedF16MinMax(source0, source1, isMax: false),
                        source2,
                        isMax: false);
                }
                else if (instruction.Opcode.StartsWith("VMax3", StringComparison.Ordinal))
                {
                    value = EmitPackedF16MinMax(
                        EmitPackedF16MinMax(source0, source1, isMax: true),
                        source2,
                        isMax: true);
                }
                else
                {
                    value = EmitVop3F16Median(source0, source1, source2);
                }

                halfResult = EmitVop3F16Result(Bitcast(_uintType, value), control);
            }
            else
            {
                var source0 = EmitVop3HalfBits(instruction, control, 0);
                var source1 = EmitVop3HalfBits(instruction, control, 1);
                var source2 = EmitVop3HalfBits(instruction, control, 2);
                var signed = instruction.Opcode.EndsWith("I16", StringComparison.Ordinal);
                var isMax = !instruction.Opcode.StartsWith("VMin3", StringComparison.Ordinal);
                if (instruction.Opcode.StartsWith("VMed3", StringComparison.Ordinal))
                {
                    halfResult = EmitVop3Integer16Median(source0, source1, source2, signed);
                }
                else
                {
                    halfResult = EmitVop3Integer16MinMax(
                        EmitVop3Integer16MinMax(source0, source1, signed, isMax),
                        source2,
                        signed,
                        isMax);
                }
            }

            var existing = LoadV(destination);
            halfResult = BitwiseAnd(halfResult, UInt(0xFFFF));
            result = (control.OperandSelect & 8) == 0
                ? BitwiseOr(BitwiseAnd(existing, UInt(0xFFFF_0000)), halfResult)
                : BitwiseOr(
                    BitwiseAnd(existing, UInt(0x0000_FFFF)),
                    ShiftLeftLogical(halfResult, UInt(16)));
            return true;
        }

        private uint EmitVop3HalfBits(
            Gen5ShaderInstruction instruction,
            Gen5Vop3Control control,
            int sourceIndex)
        {
            var raw = GetRawSource(instruction, sourceIndex);
            if (instruction.Sources[sourceIndex] is
                { Kind: Gen5OperandKind.EncodedConstant, Value: >= 240 and <= 248 } constant &&
                Gen5InlineConstants.TryDecode(constant.Value, out var floatBits))
            {
                // Floating inline constants are converted by the hardware to the
                // expected operand width. Integer inline constants remain raw bits.
                var half = (Half)BitConverter.UInt32BitsToSingle(floatBits);
                raw = UInt(BitConverter.HalfToUInt16Bits(half));
            }

            if ((control.OperandSelect & (1u << sourceIndex)) != 0)
            {
                raw = ShiftRightLogical(raw, UInt(16));
            }

            return BitwiseAnd(raw, UInt(0xFFFF));
        }

        private uint EmitVop3F16Operand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3Control control,
            int sourceIndex)
        {
            var half = EmitVop3HalfBits(instruction, control, sourceIndex);
            if ((control.AbsoluteMask & (1u << sourceIndex)) != 0)
            {
                half = BitwiseAnd(half, UInt(0x7FFF));
            }

            if ((control.NegateMask & (1u << sourceIndex)) != 0)
            {
                half = BitwiseXor(half, UInt(0x8000));
            }

            return Bitcast(_floatType, EmitHalfToFloat(half));
        }

        private uint EmitVop3F16Result(uint valueBits, Gen5Vop3Control control)
        {
            var value = Bitcast(_floatType, valueBits);
            value = control.OutputModifier switch
            {
                1 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(2)),
                2 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(4)),
                3 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(0.5f)),
                _ => value,
            };
            valueBits = Bitcast(_uintType, value);
            if (control.Clamp)
            {
                valueBits = EmitClampToUnitInterval(valueBits);
            }

            return EmitFloatToHalf(valueBits);
        }

        private uint EmitVop3F16Median(uint source0, uint source1, uint source2)
        {
            var min3 = EmitPackedF16MinMax(
                EmitPackedF16MinMax(source0, source1, isMax: false),
                source2,
                isMax: false);
            var max01 = EmitPackedF16MinMax(source0, source1, isMax: true);
            var max3 = EmitPackedF16MinMax(max01, source2, isMax: true);
            var max12 = EmitPackedF16MinMax(source1, source2, isMax: true);
            var max02 = EmitPackedF16MinMax(source0, source2, isMax: true);
            var median = _module.AddInstruction(
                SpirvOp.Select,
                _floatType,
                _module.AddInstruction(SpirvOp.FOrdEqual, _boolType, max3, source0),
                max12,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    _module.AddInstruction(SpirvOp.FOrdEqual, _boolType, max3, source1),
                    max02,
                    max01));
            var anyNan = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                _module.AddInstruction(SpirvOp.IsNan, _boolType, source0),
                _module.AddInstruction(
                    SpirvOp.LogicalOr,
                    _boolType,
                    _module.AddInstruction(SpirvOp.IsNan, _boolType, source1),
                    _module.AddInstruction(SpirvOp.IsNan, _boolType, source2)));
            return _module.AddInstruction(SpirvOp.Select, _floatType, anyNan, min3, median);
        }

        private uint EmitVop3Integer16MinMax(
            uint left,
            uint right,
            bool signed,
            bool isMax)
        {
            uint condition;
            if (signed)
            {
                var leftSigned = Bitcast(_intType, ShiftLeftLogical(left, UInt(16)));
                var rightSigned = Bitcast(_intType, ShiftLeftLogical(right, UInt(16)));
                condition = _module.AddInstruction(
                    isMax ? SpirvOp.SGreaterThan : SpirvOp.SLessThan,
                    _boolType,
                    leftSigned,
                    rightSigned);
            }
            else
            {
                condition = _module.AddInstruction(
                    isMax ? SpirvOp.UGreaterThan : SpirvOp.ULessThan,
                    _boolType,
                    left,
                    right);
            }

            return SelectU(condition, left, right);
        }

        private uint EmitVop3Integer16Median(
            uint source0,
            uint source1,
            uint source2,
            bool signed)
        {
            var max01 = EmitVop3Integer16MinMax(source0, source1, signed, isMax: true);
            var max3 = EmitVop3Integer16MinMax(max01, source2, signed, isMax: true);
            var max12 = EmitVop3Integer16MinMax(source1, source2, signed, isMax: true);
            var max02 = EmitVop3Integer16MinMax(source0, source2, signed, isMax: true);
            var equals0 = _module.AddInstruction(SpirvOp.IEqual, _boolType, max3, source0);
            var equals1 = _module.AddInstruction(SpirvOp.IEqual, _boolType, max3, source1);
            return SelectU(equals0, max12, SelectU(equals1, max02, max01));
        }

        private bool TryEmitVop3Integer16(
            Gen5ShaderInstruction instruction,
            uint destination,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3Control control)
            {
                error = $"missing vop3 control for {instruction.Opcode}";
                return false;
            }

            var left = EmitVop3HalfBits(instruction, control, 0);
            var right = EmitVop3HalfBits(instruction, control, 1);
            uint halfResult;
            switch (instruction.Opcode)
            {
                case "VAddNcU16":
                {
                    var sum = IAdd(left, right);
                    halfResult = control.Clamp
                        ? SelectU(
                            UCmp(SpirvOp.UGreaterThan, sum, UInt(0xFFFF)),
                            UInt(0xFFFF),
                            sum)
                        : sum;
                    break;
                }
                case "VSubNcU16":
                {
                    var difference = ISubU(left, right);
                    halfResult = control.Clamp
                        ? SelectU(UCmp(SpirvOp.ULessThan, left, right), UInt(0), difference)
                        : difference;
                    break;
                }
                case "VMulLoU16":
                {
                    var product = _module.AddInstruction(SpirvOp.IMul, _uintType, left, right);
                    halfResult = control.Clamp
                        ? SelectU(
                            UCmp(SpirvOp.UGreaterThan, product, UInt(0xFFFF)),
                            UInt(0xFFFF),
                            product)
                        : product;
                    break;
                }
                case "VLshrrevB16":
                    halfResult = ShiftRightLogical(right, BitwiseAnd(left, UInt(15)));
                    break;
                case "VLshlrevB16":
                    halfResult = ShiftLeftLogical(right, BitwiseAnd(left, UInt(15)));
                    break;
                case "VAshrrevI16":
                {
                    var signedRight = Bitcast(
                        _intType,
                        ShiftRightArithmetic(ShiftLeftLogical(right, UInt(16)), UInt(16)));
                    halfResult = Bitcast(
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.ShiftRightArithmetic,
                            _intType,
                            signedRight,
                            BitwiseAnd(left, UInt(15))));
                    break;
                }
                case "VMaxU16":
                case "VMinU16":
                    halfResult = EmitVop3Integer16MinMax(
                        left,
                        right,
                        signed: false,
                        isMax: instruction.Opcode == "VMaxU16");
                    break;
                case "VMaxI16":
                case "VMinI16":
                    halfResult = EmitVop3Integer16MinMax(
                        left,
                        right,
                        signed: true,
                        isMax: instruction.Opcode == "VMaxI16");
                    break;
                case "VAddNcI16":
                case "VSubNcI16":
                {
                    var signedLeft = Bitcast(
                        _intType,
                        ShiftRightArithmetic(ShiftLeftLogical(left, UInt(16)), UInt(16)));
                    var signedRight = Bitcast(
                        _intType,
                        ShiftRightArithmetic(ShiftLeftLogical(right, UInt(16)), UInt(16)));
                    var value = _module.AddInstruction(
                        instruction.Opcode == "VAddNcI16" ? SpirvOp.IAdd : SpirvOp.ISub,
                        _intType,
                        signedLeft,
                        signedRight);
                    if (control.Clamp)
                    {
                        var minimum = Bitcast(_intType, UInt(0xFFFF_8000));
                        var maximum = Bitcast(_intType, UInt(0x0000_7FFF));
                        value = _module.AddInstruction(
                            SpirvOp.Select,
                            _intType,
                            _module.AddInstruction(
                                SpirvOp.SLessThan,
                                _boolType,
                                value,
                                minimum),
                            minimum,
                            _module.AddInstruction(
                                SpirvOp.Select,
                                _intType,
                                _module.AddInstruction(
                                    SpirvOp.SGreaterThan,
                                    _boolType,
                                    value,
                                    maximum),
                                maximum,
                                value));
                    }

                    halfResult = Bitcast(_uintType, value);
                    break;
                }
                case "VMadU16":
                {
                    var addend = EmitVop3HalfBits(instruction, control, 2);
                    var value = IAdd(
                        _module.AddInstruction(SpirvOp.IMul, _uintType, left, right),
                        addend);
                    halfResult = control.Clamp
                        ? SelectU(
                            UCmp(SpirvOp.UGreaterThan, value, UInt(0xFFFF)),
                            UInt(0xFFFF),
                            value)
                        : value;
                    break;
                }
                case "VMadI16":
                {
                    uint Signed16(uint bits) => Bitcast(
                        _intType,
                        ShiftRightArithmetic(ShiftLeftLogical(bits, UInt(16)), UInt(16)));
                    var value = _module.AddInstruction(
                        SpirvOp.IAdd,
                        _intType,
                        _module.AddInstruction(
                            SpirvOp.IMul,
                            _intType,
                            Signed16(left),
                            Signed16(right)),
                        Signed16(EmitVop3HalfBits(instruction, control, 2)));
                    if (control.Clamp)
                    {
                        var minimum = Bitcast(_intType, UInt(0xFFFF_8000));
                        var maximum = Bitcast(_intType, UInt(0x0000_7FFF));
                        value = _module.AddInstruction(
                            SpirvOp.Select,
                            _intType,
                            _module.AddInstruction(SpirvOp.SLessThan, _boolType, value, minimum),
                            minimum,
                            _module.AddInstruction(
                                SpirvOp.Select,
                                _intType,
                                _module.AddInstruction(SpirvOp.SGreaterThan, _boolType, value, maximum),
                                maximum,
                                value));
                    }

                    halfResult = Bitcast(_uintType, value);
                    break;
                }
                default:
                    error = $"unsupported vop3 i16 operation {instruction.Opcode}";
                    return false;
            }

            halfResult = BitwiseAnd(halfResult, UInt(0xFFFF));
            var existing = LoadV(destination);
            result = (control.OperandSelect & 8) == 0
                ? BitwiseOr(BitwiseAnd(existing, UInt(0xFFFF_0000)), halfResult)
                : BitwiseOr(
                    BitwiseAnd(existing, UInt(0x0000_FFFF)),
                    ShiftLeftLogical(halfResult, UInt(16)));
            return true;
        }

        private bool TryEmitDivFixupF16(
            Gen5ShaderInstruction instruction,
            uint destination,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3Control control)
            {
                error = "missing vop3 control for VDivFixupF16";
                return false;
            }

            uint SourceBits(int index)
            {
                var bits = EmitVop3HalfBits(instruction, control, index);
                if ((control.AbsoluteMask & (1u << index)) != 0)
                {
                    bits = BitwiseAnd(bits, UInt(0x7FFF));
                }

                if ((control.NegateMask & (1u << index)) != 0)
                {
                    bits = BitwiseXor(bits, UInt(0x8000));
                }

                return bits;
            }

            uint IsNan16(uint bits) => _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                Equal(BitwiseAnd(bits, UInt(0x7C00)), 0x7C00),
                IsNotZero(BitwiseAnd(bits, UInt(0x03FF))));
            uint Both(uint leftCondition, uint rightCondition) =>
                _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    leftCondition,
                    rightCondition);

            var quotient = SourceBits(0);
            var denominator = SourceBits(1);
            var numerator = SourceBits(2);
            var denominatorAbs = BitwiseAnd(denominator, UInt(0x7FFF));
            var numeratorAbs = BitwiseAnd(numerator, UInt(0x7FFF));
            var sign = BitwiseAnd(BitwiseXor(denominator, numerator), UInt(0x8000));
            var denominatorZero = Equal(denominatorAbs, 0);
            var numeratorZero = Equal(numeratorAbs, 0);
            var denominatorInfinity = Equal(denominatorAbs, 0x7C00);
            var numeratorInfinity = Equal(numeratorAbs, 0x7C00);
            var invalid = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                Both(denominatorZero, numeratorZero),
                Both(denominatorInfinity, numeratorInfinity));
            var infinityResult = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                denominatorZero,
                numeratorInfinity);
            var zeroResult = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                denominatorInfinity,
                numeratorZero);
            var fixedBits = BitwiseOr(sign, BitwiseAnd(quotient, UInt(0x7FFF)));
            fixedBits = SelectU(zeroResult, sign, fixedBits);
            fixedBits = SelectU(infinityResult, BitwiseOr(sign, UInt(0x7C00)), fixedBits);
            fixedBits = SelectU(invalid, UInt(0xFE00), fixedBits);
            fixedBits = SelectU(
                IsNan16(denominator),
                BitwiseOr(denominator, UInt(0x0200)),
                fixedBits);
            fixedBits = SelectU(
                IsNan16(numerator),
                BitwiseOr(numerator, UInt(0x0200)),
                fixedBits);

            var existing = LoadV(destination);
            result = (control.OperandSelect & 8) == 0
                ? BitwiseOr(BitwiseAnd(existing, UInt(0xFFFF_0000)), fixedBits)
                : BitwiseOr(
                    BitwiseAnd(existing, UInt(0x0000_FFFF)),
                    ShiftLeftLogical(fixedBits, UInt(16)));
            return true;
        }

        // Packed 16-bit integer arithmetic (VOP3P opcodes 0x00-0x0d). Source
        // selection and negation are independent for the low and high result lanes.
        // Integer negation is two's-complement modulo 2^16. Without CLAMP the low
        // 16 result bits are retained; with CLAMP arithmetic is saturated to the
        // operation's signed or unsigned 16-bit domain.
        private bool TryEmitPackedInteger16(
            Gen5ShaderInstruction instruction,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                error = $"missing vop3p control for {instruction.Opcode}";
                return false;
            }

            uint EmitLane(bool highLane)
            {
                uint SourceBits(int index)
                {
                    var raw = GetRawSource(instruction, index);
                    var selectMask = highLane ? control.OpSelHiMask : control.OpSelMask;
                    var bits = ((selectMask >> index) & 1) != 0
                        ? ShiftRightLogical(raw, UInt(16))
                        : raw;
                    bits = BitwiseAnd(bits, UInt(0xFFFF));
                    var negateMask = highLane ? control.NegHiMask : control.NegLoMask;
                    return ((negateMask >> index) & 1) != 0
                        ? BitwiseAnd(ISubU(UInt(0), bits), UInt(0xFFFF))
                        : bits;
                }

                uint Signed(uint bits) => Bitcast(
                    _intType,
                    ShiftRightArithmetic(ShiftLeftLogical(bits, UInt(16)), UInt(16)));
                uint UnsignedClamp(uint value)
                {
                    if (!control.Clamp)
                    {
                        return value;
                    }

                    return _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.UGreaterThan,
                            _boolType,
                            value,
                            UInt(0xFFFF)),
                        UInt(0xFFFF),
                        value);
                }

                uint SignedClamp(uint value)
                {
                    if (!control.Clamp)
                    {
                        return value;
                    }

                    var minimum = Bitcast(_intType, UInt(0xFFFF_8000));
                    var maximum = Bitcast(_intType, UInt(0x0000_7FFF));
                    var lowerBounded = _module.AddInstruction(
                        SpirvOp.Select,
                        _intType,
                        _module.AddInstruction(
                            SpirvOp.SLessThan,
                            _boolType,
                            value,
                            minimum),
                        minimum,
                        value);
                    return _module.AddInstruction(
                        SpirvOp.Select,
                        _intType,
                        _module.AddInstruction(
                            SpirvOp.SGreaterThan,
                            _boolType,
                            lowerBounded,
                            maximum),
                        maximum,
                        lowerBounded);
                }

                var source0 = SourceBits(0);
                var source1 = SourceBits(1);
                var source2 = SourceBits(2);
                uint lane;
                switch (instruction.Opcode)
                {
                    case "VPkMadI16":
                        lane = SignedClamp(_module.AddInstruction(
                            SpirvOp.IAdd,
                            _intType,
                            _module.AddInstruction(
                                SpirvOp.IMul,
                                _intType,
                                Signed(source0),
                                Signed(source1)),
                            Signed(source2)));
                        lane = Bitcast(_uintType, lane);
                        break;
                    case "VPkAddI16":
                    case "VPkSubI16":
                        lane = SignedClamp(_module.AddInstruction(
                            instruction.Opcode == "VPkAddI16" ? SpirvOp.IAdd : SpirvOp.ISub,
                            _intType,
                            Signed(source0),
                            Signed(source1)));
                        lane = Bitcast(_uintType, lane);
                        break;
                    case "VPkAshrrevI16":
                        lane = Bitcast(
                            _uintType,
                            _module.AddInstruction(
                                SpirvOp.ShiftRightArithmetic,
                                _intType,
                                Signed(source1),
                                BitwiseAnd(source0, UInt(15))));
                        break;
                    case "VPkMaxI16":
                    case "VPkMinI16":
                    {
                        var left = Signed(source0);
                        var right = Signed(source1);
                        var compare = instruction.Opcode == "VPkMaxI16"
                            ? SpirvOp.SGreaterThanEqual
                            : SpirvOp.SLessThan;
                        lane = Bitcast(
                            _uintType,
                            _module.AddInstruction(
                                SpirvOp.Select,
                                _intType,
                                _module.AddInstruction(compare, _boolType, left, right),
                                left,
                                right));
                        break;
                    }
                    case "VPkMulLoU16":
                        lane = UnsignedClamp(_module.AddInstruction(
                            SpirvOp.IMul,
                            _uintType,
                            source0,
                            source1));
                        break;
                    case "VPkLshlrevB16":
                        lane = ShiftLeftLogical(source1, BitwiseAnd(source0, UInt(15)));
                        break;
                    case "VPkLshrrevB16":
                        lane = ShiftRightLogical(source1, BitwiseAnd(source0, UInt(15)));
                        break;
                    case "VPkMadU16":
                        lane = UnsignedClamp(IAdd(
                            _module.AddInstruction(
                                SpirvOp.IMul,
                                _uintType,
                                source0,
                                source1),
                            source2));
                        break;
                    case "VPkAddU16":
                        lane = UnsignedClamp(IAdd(source0, source1));
                        break;
                    case "VPkSubU16":
                        lane = control.Clamp
                            ? _module.AddInstruction(
                                SpirvOp.Select,
                                _uintType,
                                _module.AddInstruction(
                                    SpirvOp.ULessThan,
                                    _boolType,
                                    source0,
                                    source1),
                                UInt(0),
                                ISubU(source0, source1))
                            : ISubU(source0, source1);
                        break;
                    case "VPkMaxU16":
                    case "VPkMinU16":
                    {
                        var compare = instruction.Opcode == "VPkMaxU16"
                            ? SpirvOp.UGreaterThanEqual
                            : SpirvOp.ULessThan;
                        lane = _module.AddInstruction(
                            SpirvOp.Select,
                            _uintType,
                            _module.AddInstruction(compare, _boolType, source0, source1),
                            source0,
                            source1);
                        break;
                    }
                    default:
                        lane = UInt(0);
                        break;
                }

                return BitwiseAnd(lane, UInt(0xFFFF));
            }

            var low = EmitLane(highLane: false);
            var high = EmitLane(highLane: true);
            result = BitwiseOr(low, ShiftLeftLogical(high, UInt(16)));
            return true;
        }

        private bool TryEmitPackedIntegerDot(
            Gen5ShaderInstruction instruction,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                error = $"missing vop3p control for {instruction.Opcode}";
                return false;
            }

            var signed = instruction.Opcode.Contains("I32I", StringComparison.Ordinal);
            var componentBits = instruction.Opcode.StartsWith("VDot2", StringComparison.Ordinal)
                ? 16
                : instruction.Opcode.StartsWith("VDot4", StringComparison.Ordinal) ? 8 : 4;
            var componentCount = 32 / componentBits;
            var componentMask = (1u << componentBits) - 1;
            var source0 = GetRawSource(instruction, 0);
            var source1 = GetRawSource(instruction, 1);

            if (signed)
            {
                uint SignedComponent(uint source, int index)
                {
                    var bits = BitwiseAnd(
                        ShiftRightLogical(source, UInt((uint)(index * componentBits))),
                        UInt(componentMask));
                    var shift = UInt((uint)(32 - componentBits));
                    return Bitcast(
                        _intType,
                        ShiftRightArithmetic(ShiftLeftLogical(bits, shift), shift));
                }

                var total = _module.AddInstruction(
                    SpirvOp.SConvert,
                    _longType,
                    Bitcast(_intType, GetRawSource(instruction, 2)));
                for (var index = 0; index < componentCount; index++)
                {
                    var left = _module.AddInstruction(
                        SpirvOp.SConvert,
                        _longType,
                        SignedComponent(source0, index));
                    var right = _module.AddInstruction(
                        SpirvOp.SConvert,
                        _longType,
                        SignedComponent(source1, index));
                    total = _module.AddInstruction(
                        SpirvOp.IAdd,
                        _longType,
                        total,
                        _module.AddInstruction(SpirvOp.IMul, _longType, left, right));
                }

                if (control.Clamp)
                {
                    var minimum = _module.Constant64(
                        _longType,
                        unchecked((ulong)(long)int.MinValue));
                    var maximum = _module.Constant64(_longType, int.MaxValue);
                    total = _module.AddInstruction(
                        SpirvOp.Select,
                        _longType,
                        _module.AddInstruction(
                            SpirvOp.SLessThan,
                            _boolType,
                            total,
                            minimum),
                        minimum,
                        total);
                    total = _module.AddInstruction(
                        SpirvOp.Select,
                        _longType,
                        _module.AddInstruction(
                            SpirvOp.SGreaterThan,
                            _boolType,
                            total,
                            maximum),
                        maximum,
                        total);
                }

                result = Bitcast(
                    _uintType,
                    _module.AddInstruction(SpirvOp.SConvert, _intType, total));
                return true;
            }

            var unsignedTotal = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                GetRawSource(instruction, 2));
            for (var index = 0; index < componentCount; index++)
            {
                uint Component(uint source) => BitwiseAnd(
                    ShiftRightLogical(source, UInt((uint)(index * componentBits))),
                    UInt(componentMask));
                var left = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    Component(source0));
                var right = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    Component(source1));
                unsignedTotal = _module.AddInstruction(
                    SpirvOp.IAdd,
                    _ulongType,
                    unsignedTotal,
                    _module.AddInstruction(SpirvOp.IMul, _ulongType, left, right));
            }

            if (control.Clamp)
            {
                var maximum = _module.Constant64(_ulongType, uint.MaxValue);
                unsignedTotal = _module.AddInstruction(
                    SpirvOp.Select,
                    _ulongType,
                    _module.AddInstruction(
                        SpirvOp.UGreaterThan,
                        _boolType,
                        unsignedTotal,
                        maximum),
                    maximum,
                    unsignedTotal);
            }

            result = _module.AddInstruction(SpirvOp.UConvert, _uintType, unsignedTotal);
            return true;
        }

        // V_DOT2_F32_F16 is the one RDNA2 dot instruction whose third source is
        // scalar f32 rather than packed data. AMD Table 86 and LLVM's
        // VOP3PModsDOT/VOP3PModsF32 selectors define two distinct modifier
        // domains: op_sel/neg_lo and op_sel_hi/neg_hi select and negate the two
        // f16 components of src0/src1, while src2 uses neg_hi[2] as fabs and
        // neg_lo[2] as fneg. The hardware also flushes f32 denormal src2 and the
        // f32 result regardless of the shader's denormal mode.
        private bool TryEmitPackedFloatDot(
            Gen5ShaderInstruction instruction,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                error = $"missing vop3p control for {instruction.Opcode}";
                return false;
            }

            var source2Bits = FlushFloat32Denormal(GetRawSource(instruction, 2));
            var source2 = Bitcast(_floatType, source2Bits);
            if ((control.NegHiMask & 4) != 0)
            {
                source2 = Ext(4, _floatType, source2);
            }

            if ((control.NegLoMask & 4) != 0)
            {
                source2 = _module.AddInstruction(SpirvOp.FNegate, _floatType, source2);
            }

            var low = Ext(
                50,
                _floatType,
                EmitPackedF16Operand(instruction, control, 0, highLane: false),
                EmitPackedF16Operand(instruction, control, 1, highLane: false),
                source2);
            var dot = Ext(
                50,
                _floatType,
                EmitPackedF16Operand(instruction, control, 0, highLane: true),
                EmitPackedF16Operand(instruction, control, 1, highLane: true),
                low);
            result = FlushFloat32Denormal(Bitcast(_uintType, dot));
            if (control.Clamp)
            {
                result = EmitClampToUnitInterval(result);
            }

            return true;
        }

        private uint FlushFloat32Denormal(uint bits)
        {
            var exponentZero = Equal(BitwiseAnd(bits, UInt(0x7F80_0000)), 0);
            var mantissaNonZero = IsNotZero(BitwiseAnd(bits, UInt(0x007F_FFFF)));
            var subnormal = _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                exponentZero,
                mantissaNonZero);
            return SelectU(subnormal, BitwiseAnd(bits, UInt(0x8000_0000)), bits);
        }

        // Packed f16 (VOP3P) arithmetic. Each source register holds two f16 values,
        // one per result lane. Every f16<->f32 conversion is done with the explicit
        // integer sequences below (EmitHalfToFloat / EmitFloatToHalf) instead of
        // GLSL UnpackHalf2x16 / PackHalf2x16, whose subnormal and rounding behaviour
        // is implementation-defined without float-controls execution modes. The two
        // lanes are computed independently: each operand half is widened exactly to
        // f32, op_sel/op_sel_hi pick the source half and neg_lo/neg_hi negate it, the
        // op runs in f32, and the result is rounded back to f16 with round-to-nearest-
        // even. For add and mul this is bit-exact to a true f16 op (the f32 result
        // rounds losslessly to f16 by the double-rounding theorem; a f16 product even
        // fits in f32 exactly). min/max carry no rounding, so they are exact once the
        // conversions are. v_pk_fma_f16 cannot be reproduced by a plain f32
        // multiply-add plus a pack (that double-rounds), so it goes through the
        // round-to-odd sequence in EmitPackedF16FusedMultiplyAdd instead.
        private bool TryEmitPackedF16(
            Gen5ShaderInstruction instruction,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                error = $"missing vop3p control for {instruction.Opcode}";
                return false;
            }

            var sourceCount = instruction.Opcode == "VPkFmaF16" ? 3 : 2;
            for (var index = 0; index < sourceCount; index++)
            {
                var source = instruction.Sources[index];
                if (source.Kind is not (Gen5OperandKind.VectorRegister or Gen5OperandKind.ScalarRegister))
                {
                    error =
                        $"unsupported vop3p operand {source} for {instruction.Opcode} (first slice: registers only)";
                    return false;
                }
            }

            var low = EmitPackedF16Lane(instruction, control, highLane: false);
            var high = EmitPackedF16Lane(instruction, control, highLane: true);
            result = BitwiseOr(low, ShiftLeftLogical(high, UInt(16)));
            return true;
        }

        // V_FMA_MIX_F32 / _MIXLO_F16 / _MIXHI_F16 (VOP3P opcodes 0x20 / 0x21 /
        // 0x22). Unlike the packed v_pk_* ops these compute a single f32
        // fma(a, b, c): each of the three sources is *independently* read as
        // either a full f32 register/constant or one f16 half widened to f32,
        // selected per operand by op_sel_hi (read as f16 when set) and op_sel
        // (which half feeds the f32). For the mix ops the VOP3P neg_hi field is
        // the absolute-value modifier and neg negates, applied abs-then-neg to
        // match the hardware and shadPS4's GetSrcMix. _MIXLO / _MIXHI round the
        // f32 result back to f16 and write it into the low / high 16 bits of
        // vdst, leaving the other half intact.
        private bool TryEmitFmaMix(
            Gen5ShaderInstruction instruction,
            uint destination,
            out uint result,
            out string error)
        {
            result = 0;
            error = string.Empty;
            if (instruction.Control is not Gen5Vop3pControl control)
            {
                error = $"missing vop3p control for {instruction.Opcode}";
                return false;
            }

            var product = Bitcast(
                _uintType,
                Ext(
                    50,
                    _floatType,
                    EmitFmaMixOperand(instruction, control, 0),
                    EmitFmaMixOperand(instruction, control, 1),
                    EmitFmaMixOperand(instruction, control, 2)));
            if (control.Clamp)
            {
                product = EmitClampToUnitInterval(product);
            }

            if (instruction.Opcode == "VFmaMixF32")
            {
                result = product;
                return true;
            }

            // _MIXLO / _MIXHI: narrow to f16 and merge into one half of vdst.
            var half = EmitFloatToHalf(product);
            var existing = LoadV(destination);
            result = instruction.Opcode == "VFmaMixloF16"
                ? BitwiseOr(BitwiseAnd(existing, UInt(0xFFFF_0000)), half)
                : BitwiseOr(
                    BitwiseAnd(existing, UInt(0x0000_FFFF)),
                    ShiftLeftLogical(half, UInt(16)));
            return true;
        }

        // Reads one V_FMA_MIX source as an f32. op_sel_hi selects whether a
        // register operand is taken as an f16 (the half picked by op_sel, widened
        // exactly to f32) or as a full f32; inline constants are always f32. The
        // per-operand neg_hi bit takes the absolute value and neg negates, in that
        // order (abs-then-neg), reusing the VOP3P modifier fields the way the mix
        // ops define them rather than the packed low/high-lane meaning.
        private uint EmitFmaMixOperand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            int index)
        {
            var source = instruction.Sources[index];
            var readAsHalf =
                ((control.OpSelHiMask >> index) & 1) != 0 &&
                source.Kind is Gen5OperandKind.VectorRegister or Gen5OperandKind.ScalarRegister;

            uint value;
            if (readAsHalf)
            {
                var raw = GetRawSource(instruction, index);
                var half = ((control.OpSelMask >> index) & 1) != 0
                    ? ShiftRightLogical(raw, UInt(16))
                    : raw;
                value = Bitcast(_floatType, EmitHalfToFloat(half));
            }
            else
            {
                value = GetFloatSource(instruction, index);
            }

            if (((control.NegHiMask >> index) & 1) != 0)
            {
                value = Ext(4, _floatType, value);
            }

            if (((control.NegLoMask >> index) & 1) != 0)
            {
                value = _module.AddInstruction(SpirvOp.FNegate, _floatType, value);
            }

            return value;
        }

        // Computes one result lane (low or high) as a packed 16-bit f16 value.
        // The op runs in f32 and its result is narrowed back to f16 exactly (see
        // EmitFloatToHalf). When the clamp modifier is set the pre-narrowing f32
        // value is saturated to [0, 1] first; because 0.0 and 1.0 are exact in both
        // f32 and f16 and the clamp is monotonic, clamping before the narrowing
        // gives the same f16 the hardware produces by clamping the f16 result. For
        // the fused multiply-add the pre-narrowing value is the round-to-odd f32
        // from EmitPackedF16FusedMultiplyAdd, and round-to-odd preserves that
        // equivalence through the final round-to-nearest-even.
        private uint EmitPackedF16Lane(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            bool highLane)
        {
            var left = EmitPackedF16Operand(instruction, control, 0, highLane);
            var right = EmitPackedF16Operand(instruction, control, 1, highLane);
            uint value;
            if (instruction.Opcode == "VPkFmaF16")
            {
                var addend = EmitPackedF16Operand(instruction, control, 2, highLane);
                value = EmitPackedF16FusedMultiplyAdd(left, right, addend);
            }
            else
            {
                value = Bitcast(_uintType, instruction.Opcode switch
                {
                    "VPkAddF16" => _module.AddInstruction(SpirvOp.FAdd, _floatType, left, right),
                    "VPkMulF16" => _module.AddInstruction(SpirvOp.FMul, _floatType, left, right),
                    "VPkMinF16" => EmitPackedF16MinMax(left, right, isMax: false),
                    "VPkMaxF16" => EmitPackedF16MinMax(left, right, isMax: true),
                    _ => left,
                });
            }

            if (control.Clamp)
            {
                value = EmitClampToUnitInterval(value);
            }

            return EmitFloatToHalf(value);
        }

        // Saturates an f32 bit pattern to [0, 1] the way the VOP3P clamp modifier
        // does: below 0 (and NaN, since the ordered compare is false for it) becomes
        // 0, above 1 becomes 1. Ordered compares match the hardware's NaN-to-zero
        // behaviour without a separate IsNan test.
        private uint EmitClampToUnitInterval(uint valueBits)
        {
            var value = Bitcast(_floatType, valueBits);
            var aboveZero = _module.AddInstruction(SpirvOp.FOrdGreaterThan, _boolType, value, Float(0));
            var lowerBounded = _module.AddInstruction(SpirvOp.Select, _floatType, aboveZero, value, Float(0));
            var belowOne = _module.AddInstruction(SpirvOp.FOrdLessThan, _boolType, lowerBounded, Float(1));
            var clamped = _module.AddInstruction(SpirvOp.Select, _floatType, belowOne, lowerBounded, Float(1));
            return Bitcast(_uintType, clamped);
        }

        // Fused f16 multiply-add with a single rounding, emulated in f32 without the
        // Float16 capability. The f32 product of two widened f16 values is exact
        // (11-bit significands, and the exponent stays inside the f32 normal range:
        // any non-zero product magnitude is in [2^-48, 2^33]), so only the addition
        // rounds. An f32 add then an f16 pack would round twice; instead the add is
        // corrected to round-to-odd, which a following round-to-nearest-even pack
        // turns into the exactly-once-rounded fused result (innocuous double rounding
        // holds because f32 carries 24 significand bits >= 11 + 2).
        //
        // sum = RN(product + addend); Knuth's 2Sum recovers the exact residual
        // (product + addend) - sum from four more RN ops. 2Sum is exact for any two
        // finite f32 inputs; no intermediate here can overflow (|product| < 2^33,
        // |addend| < 2^16) and none can enter the f32 subnormal range (every finite
        // value in play is a multiple of 2^-48 by construction), so implementation
        // f32 denorm-flush modes never see a denormal. If the residual says the sum
        // was inexact and the sum's significand is even, step one ulp towards the
        // true value: consecutive floats have consecutive sign-magnitude encodings,
        // so that neighbour is the enclosing float with the odd significand.
        //
        // Inf/NaN inputs make the residual NaN (e.g. sum - addend = Inf - Inf); the
        // ordered compare below is then false and the IEEE sum passes through
        // unchanged. A residual of zero also covers the exact-sum case, where the
        // parity fix must not fire. Returns the round-to-odd f32 bit pattern.
        private uint EmitPackedF16FusedMultiplyAdd(uint left, uint right, uint addend)
        {
            var product = EmitPreciseFloat(SpirvOp.FMul, left, right);
            var sum = EmitPreciseFloat(SpirvOp.FAdd, product, addend);

            var productPart = EmitPreciseFloat(SpirvOp.FSub, sum, addend);
            var addendPart = EmitPreciseFloat(SpirvOp.FSub, sum, productPart);
            var productError = EmitPreciseFloat(SpirvOp.FSub, product, productPart);
            var addendError = EmitPreciseFloat(SpirvOp.FSub, addend, addendPart);
            var residual = EmitPreciseFloat(SpirvOp.FAdd, productError, addendError);

            var sumBits = Bitcast(_uintType, sum);
            var residualBits = Bitcast(_uintType, residual);
            var inexact = _module.AddInstruction(
                SpirvOp.FOrdNotEqual, _boolType, residual, Float(0));
            var evenSignificand = Equal(BitwiseAnd(sumBits, UInt(1)), 0);
            var adjust = _module.AddInstruction(
                SpirvOp.LogicalAnd, _boolType, inexact, evenSignificand);

            // Residual sign relative to the sum picks the step direction: same sign
            // means the true value lies away from zero (encoding + 1), opposite sign
            // means towards zero (encoding - 1). The sum cannot be zero here (any
            // inexact sum has magnitude >= 2^-48) and cannot be the largest finite
            // value (its significand is odd), so the step never crosses zero or Inf.
            var towardZero = IsNotZero(
                BitwiseAnd(BitwiseXor(sumBits, residualBits), UInt(0x8000_0000)));
            var stepped = SelectU(
                towardZero,
                ISubU(sumBits, UInt(1)),
                IAdd(sumBits, UInt(1)));
            return SelectU(adjust, stepped, sumBits);
        }

        // A float op the driver must evaluate exactly as written. The 2Sum
        // residual above is error-free only op by op; without NoContraction
        // driver compilers fold the sequence (e.g. contract product+sum into an
        // f32 fma and simplify the rebuilt terms), collapsing the residual to
        // zero. Observed on AMD RDNA3 Windows: the pinned midpoint case decays
        // to the double-rounded result unless every op in the chain is marked.
        private uint EmitPreciseFloat(SpirvOp operation, uint left, uint right)
        {
            var value = _module.AddInstruction(operation, _floatType, left, right);
            _module.AddDecoration(value, SpirvDecoration.NoContraction);
            return value;
        }

        // Reads source `index`, selects the half feeding this lane (op_sel / op_sel_hi),
        // widens it exactly to f32 and applies the lane's negate modifier (neg_lo / neg_hi).
        private uint EmitPackedF16Operand(
            Gen5ShaderInstruction instruction,
            Gen5Vop3pControl control,
            int index,
            bool highLane)
        {
            var raw = GetRawSource(instruction, index);
            var selectMask = highLane ? control.OpSelHiMask : control.OpSelMask;
            var half = ((selectMask >> index) & 1) != 0
                ? ShiftRightLogical(raw, UInt(16))
                : raw;
            var value = Bitcast(_floatType, EmitHalfToFloat(half));
            var negateMask = highLane ? control.NegHiMask : control.NegLoMask;
            if (((negateMask >> index) & 1) != 0)
            {
                value = _module.AddInstruction(SpirvOp.FNegate, _floatType, value);
            }

            return value;
        }

        // AMD's f16 min/max is minNum/maxNum-like for NaNs and has an explicit
        // signed-zero rule: min(+0,-0) is -0 and max(+0,-0) is +0. Preserve that
        // rule here instead of leaving the choice to an unordered host comparison.
        private uint EmitPackedF16MinMax(uint left, uint right, bool isMax)
        {
            var compare = _module.AddInstruction(
                isMax ? SpirvOp.FOrdGreaterThan : SpirvOp.FOrdLessThan,
                _boolType,
                left,
                right);
            var numeric = _module.AddInstruction(
                SpirvOp.Select, _floatType, compare, left, right);
            var leftNan = _module.AddInstruction(SpirvOp.IsNan, _boolType, left);
            var rightNan = _module.AddInstruction(SpirvOp.IsNan, _boolType, right);
            var withRight = _module.AddInstruction(
                SpirvOp.Select, _floatType, rightNan, left, numeric);
            var withoutNan = _module.AddInstruction(
                SpirvOp.Select, _floatType, leftNan, right, withRight);
            var leftBits = Bitcast(_uintType, left);
            var rightBits = Bitcast(_uintType, right);
            var leftZero = Equal(BitwiseAnd(leftBits, UInt(0x7FFF_FFFF)), 0);
            var rightZero = Equal(BitwiseAnd(rightBits, UInt(0x7FFF_FFFF)), 0);
            var bothZero = _module.AddInstruction(
                SpirvOp.LogicalAnd, _boolType, leftZero, rightZero);
            var zeroBits = isMax
                ? BitwiseAnd(leftBits, rightBits)
                : BitwiseOr(leftBits, rightBits);
            return _module.AddInstruction(
                SpirvOp.Select,
                _floatType,
                bothZero,
                Bitcast(_floatType, zeroBits),
                withoutNan);
        }

        // Widens an f16 value held in the low 16 bits of `halfBits` to an f32 bit
        // pattern, exactly (subnormals normalised, Inf/NaN and signed zero preserved).
        // Mirrors the branchless HalfToFloat reference validated against System.Half.
        private uint EmitHalfToFloat(uint halfBits)
        {
            var sign = ShiftLeftLogical(BitwiseAnd(halfBits, UInt(0x8000)), UInt(16));
            var exponent = BitwiseAnd(ShiftRightLogical(halfBits, UInt(10)), UInt(0x1F));
            var mantissa = BitwiseAnd(halfBits, UInt(0x3FF));

            var normal = BitwiseOr(
                ShiftLeftLogical(IAdd(exponent, UInt(112)), UInt(23)),
                ShiftLeftLogical(mantissa, UInt(13)));
            var infinityNan = BitwiseOr(UInt(0x7F80_0000), ShiftLeftLogical(mantissa, UInt(13)));

            // Subnormal: normalise the mantissa. FindUMsb of (mantissa | 1) keeps the
            // op defined when mantissa is 0; that lane is discarded by the select below.
            var highBit = Ext(75, _uintType, BitwiseOr(mantissa, UInt(1)));
            var shift = ISubU(UInt(23), highBit);
            var subFraction = BitwiseAnd(ShiftLeftLogical(mantissa, shift), UInt(0x7F_FFFF));
            var subnormal = SelectU(
                IsNotZero(mantissa),
                BitwiseOr(ShiftLeftLogical(IAdd(highBit, UInt(103)), UInt(23)), subFraction),
                UInt(0));

            var magnitude = SelectU(
                Equal(exponent, 0),
                subnormal,
                SelectU(Equal(exponent, 31), infinityNan, normal));
            return BitwiseOr(sign, magnitude);
        }

        // Narrows an f32 bit pattern to an f16 value in the low 16 bits, rounding to
        // nearest even (subnormals, overflow-to-Inf and NaN/Inf handled). Mirrors the
        // branchless FloatToHalf reference validated exhaustively against System.Half.
        private uint EmitFloatToHalf(uint bits)
        {
            var sign = BitwiseAnd(ShiftRightLogical(bits, UInt(16)), UInt(0x8000));
            var absolute = BitwiseAnd(bits, UInt(0x7FFF_FFFF));

            var isInfinityNan = UCmp(SpirvOp.UGreaterThanEqual, absolute, UInt(0x7F80_0000));
            var isNan = UCmp(SpirvOp.UGreaterThan, absolute, UInt(0x7F80_0000));
            var infinityNan = BitwiseOr(
                BitwiseOr(sign, UInt(0x7C00)),
                SelectU(isNan, UInt(0x200), UInt(0)));

            var exponent = ShiftRightLogical(absolute, UInt(23));
            var mantissa = BitwiseAnd(absolute, UInt(0x7F_FFFF));
            var significand = BitwiseOr(mantissa, UInt(0x80_0000));

            // Normal path: round the 24-bit significand down to 11 bits (>> 13) with
            // round-to-nearest-even; the carry folds naturally into the exponent.
            var roundBit = BitwiseAnd(ShiftRightLogical(significand, UInt(13)), UInt(1));
            var rounded = ShiftRightLogical(IAdd(IAdd(significand, UInt(0xFFF)), roundBit), UInt(13));
            var halfExponent = ISubU(exponent, UInt(112));
            var normalBits = IAdd(ShiftLeftLogical(halfExponent, UInt(10)), ISubU(rounded, UInt(0x400)));
            var normal = SelectU(
                UCmp(SpirvOp.UGreaterThanEqual, exponent, UInt(113)),
                SelectU(UCmp(SpirvOp.UGreaterThanEqual, normalBits, UInt(0x7C00)), UInt(0x7C00), normalBits),
                UInt(0));

            // Subnormal path: value = round(significand >> (126 - exponent)) with RNE.
            // The shift is clamped to 25 so it stays defined; on this path it is >= 14.
            var distance = ISubU(UInt(126), exponent);
            var shift = SelectU(UCmp(SpirvOp.UGreaterThan, distance, UInt(25)), UInt(25), distance);
            var shiftMask = ISubU(ShiftLeftLogical(UInt(1), shift), UInt(1));
            var halfWay = ShiftLeftLogical(UInt(1), ISubU(shift, UInt(1)));
            var lowBits = BitwiseAnd(significand, shiftMask);
            var quotient = ShiftRightLogical(significand, shift);
            var roundUp = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                UCmp(SpirvOp.UGreaterThan, lowBits, halfWay),
                _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    Equal(lowBits, halfWay),
                    IsNotZero(BitwiseAnd(quotient, UInt(1)))));
            var subnormal = IAdd(quotient, SelectU(roundUp, UInt(1), UInt(0)));

            var isSubnormal = UCmp(SpirvOp.ULessThanEqual, exponent, UInt(112));
            var finite = SelectU(isSubnormal, subnormal, normal);
            return SelectU(isInfinityNan, infinityNan, BitwiseOr(sign, finite));
        }

        private uint SelectU(uint condition, uint whenTrue, uint whenFalse) =>
            _module.AddInstruction(SpirvOp.Select, _uintType, condition, whenTrue, whenFalse);

        private uint UCmp(SpirvOp operation, uint left, uint right) =>
            _module.AddInstruction(operation, _boolType, left, right);

        private uint Equal(uint value, uint constant) =>
            _module.AddInstruction(SpirvOp.IEqual, _boolType, value, UInt(constant));

        private uint ISubU(uint left, uint right) =>
            _module.AddInstruction(SpirvOp.ISub, _uintType, left, right);

        private bool TryEmitVectorCompare(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            uint condition = _module.ConstantBool(false);
            var opcode = instruction.Opcode;
            if (opcode.EndsWith("F64", StringComparison.Ordinal))
            {
                condition = EmitFloat64Compare(instruction);
            }
            else if (opcode is
                "VCmpClassF32" or "VCmpxClassF32" or
                "VCmpClassF16" or "VCmpxClassF16")
            {
                var half = opcode.EndsWith("F16", StringComparison.Ordinal);
                var source = half
                    ? EmitScalarF16Operand(instruction, 0)
                    : GetFloatSource(instruction, 0);
                var raw = half
                    ? EmitScalarF16SourceBits(instruction, 0)
                    : GetRawSource(instruction, 0);
                var mask = GetRawSource(instruction, 1);
                var negative = IsNotZero(BitwiseAnd(raw, UInt(half ? 0x8000u : 0x8000_0000u)));
                var positive = _module.AddInstruction(
                    SpirvOp.LogicalNot,
                    _boolType,
                    negative);
                var nan = _module.AddInstruction(SpirvOp.IsNan, _boolType, source);
                var infinity =
                    _module.AddInstruction(SpirvOp.IsInf, _boolType, source);
                var zero = _module.AddInstruction(
                    SpirvOp.FOrdEqual,
                    _boolType,
                    source,
                    Float(0));
                var absolute = Ext(4, _floatType, source);
                var nonzero = _module.AddInstruction(
                    SpirvOp.FOrdGreaterThan,
                    _boolType,
                    absolute,
                    Float(0));
                var belowNormal = _module.AddInstruction(
                    SpirvOp.FOrdLessThan,
                    _boolType,
                    absolute,
                    Bitcast(_floatType, UInt(half ? 0x3880_0000u : 0x0080_0000u)));
                var subnormal = _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    nonzero,
                    belowNormal);
                var special = _module.AddInstruction(
                    SpirvOp.LogicalOr,
                    _boolType,
                    nan,
                    _module.AddInstruction(
                        SpirvOp.LogicalOr,
                        _boolType,
                        infinity,
                        _module.AddInstruction(
                            SpirvOp.LogicalOr,
                            _boolType,
                            zero,
                            subnormal)));
                var normal = _module.AddInstruction(
                    SpirvOp.LogicalNot,
                    _boolType,
                    special);

                uint MaskedClass(uint bits, uint value)
                {
                    var enabled = IsNotZero(BitwiseAnd(mask, UInt(bits)));
                    return _module.AddInstruction(
                        SpirvOp.LogicalAnd,
                        _boolType,
                        enabled,
                        value);
                }

                uint SignedClass(uint negativeBit, uint positiveBit, uint value)
                {
                    var negativeClass = MaskedClass(
                        negativeBit,
                        _module.AddInstruction(
                            SpirvOp.LogicalAnd,
                            _boolType,
                            negative,
                            value));
                    var positiveClass = MaskedClass(
                        positiveBit,
                        _module.AddInstruction(
                            SpirvOp.LogicalAnd,
                            _boolType,
                            positive,
                            value));
                    return _module.AddInstruction(
                        SpirvOp.LogicalOr,
                        _boolType,
                        negativeClass,
                        positiveClass);
                }

                condition = MaskedClass(0x003, nan);
                condition = _module.AddInstruction(
                    SpirvOp.LogicalOr,
                    _boolType,
                    condition,
                    SignedClass(0x004, 0x200, infinity));
                condition = _module.AddInstruction(
                    SpirvOp.LogicalOr,
                    _boolType,
                    condition,
                    SignedClass(0x008, 0x100, normal));
                condition = _module.AddInstruction(
                    SpirvOp.LogicalOr,
                    _boolType,
                    condition,
                    SignedClass(0x010, 0x080, subnormal));
                condition = _module.AddInstruction(
                    SpirvOp.LogicalOr,
                    _boolType,
                    condition,
                    SignedClass(0x020, 0x040, zero));
            }
            else if (opcode is
                     "VCmpFF32" or "VCmpxFF32" or
                     "VCmpFF16" or "VCmpxFF16" or
                     "VCmpFI32" or "VCmpxFI32" or
                     "VCmpFU32" or "VCmpxFU32")
            {
                condition = _module.ConstantBool(false);
            }
            else if (opcode is
                     "VCmpTruF32" or "VCmpxTruF32" or
                     "VCmpTruF16" or "VCmpxTruF16" or
                     "VCmpTI32" or "VCmpxTI32" or
                     "VCmpTU32" or "VCmpxTU32")
            {
                condition = _module.ConstantBool(true);
            }
            else if (opcode is
                      "VCmpOF32" or "VCmpxOF32" or
                     "VCmpUF32" or "VCmpxUF32" or
                     "VCmpOF16" or "VCmpxOF16" or
                     "VCmpUF16" or "VCmpxUF16")
            {
                var half = opcode.EndsWith("F16", StringComparison.Ordinal);
                var left = half
                    ? EmitScalarF16Operand(instruction, 0)
                    : GetFloatSource(instruction, 0);
                var right = half
                    ? EmitScalarF16Operand(instruction, 1)
                    : GetFloatSource(instruction, 1);
                var unordered = _module.AddInstruction(
                    SpirvOp.LogicalOr,
                    _boolType,
                    _module.AddInstruction(SpirvOp.IsNan, _boolType, left),
                    _module.AddInstruction(SpirvOp.IsNan, _boolType, right));
                condition = opcode is
                    "VCmpUF32" or "VCmpxUF32" or
                    "VCmpUF16" or "VCmpxUF16"
                    ? unordered
                     : _module.AddInstruction(SpirvOp.LogicalNot, _boolType, unordered);
            }
            else if (opcode.EndsWith("I64", StringComparison.Ordinal) ||
                     opcode.EndsWith("U64", StringComparison.Ordinal))
            {
                var signed = opcode.EndsWith("I64", StringComparison.Ordinal);
                var left = GetRawSource64(instruction, 0);
                var right = GetRawSource64(instruction, 1);
                if (signed)
                {
                    left = Bitcast(_longType, left);
                    right = Bitcast(_longType, right);
                }

                var operation = TrimCompareOpcode(opcode) switch
                {
                    "Eq" => SpirvOp.IEqual,
                    "Ne" => SpirvOp.INotEqual,
                    "Lt" => signed ? SpirvOp.SLessThan : SpirvOp.ULessThan,
                    "Le" => signed ? SpirvOp.SLessThanEqual : SpirvOp.ULessThanEqual,
                    "Gt" => signed ? SpirvOp.SGreaterThan : SpirvOp.UGreaterThan,
                    "Ge" => signed ? SpirvOp.SGreaterThanEqual : SpirvOp.UGreaterThanEqual,
                    _ => SpirvOp.Nop,
                };
                if (operation == SpirvOp.Nop)
                {
                    condition = TrimCompareOpcode(opcode) switch
                    {
                        "F" => _module.ConstantBool(false),
                        "T" => _module.ConstantBool(true),
                        _ => 0,
                    };
                    if (condition == 0)
                    {
                        error = $"unsupported integer 64-bit compare {opcode}";
                        return false;
                    }
                }
                else
                {
                    condition = _module.AddInstruction(operation, _boolType, left, right);
                }
            }
            else if (opcode.EndsWith("F32", StringComparison.Ordinal) ||
                     opcode.EndsWith("F16", StringComparison.Ordinal))
            {
                var half = opcode.EndsWith("F16", StringComparison.Ordinal);
                var left = half
                    ? EmitScalarF16Operand(instruction, 0)
                    : GetFloatSource(instruction, 0);
                var right = half
                    ? EmitScalarF16Operand(instruction, 1)
                    : GetFloatSource(instruction, 1);
                var operation = TrimCompareOpcode(opcode) switch
                {
                    "Lt" => SpirvOp.FOrdLessThan,
                    "Eq" => SpirvOp.FOrdEqual,
                    "Le" => SpirvOp.FOrdLessThanEqual,
                    "Gt" => SpirvOp.FOrdGreaterThan,
                    "Lg" => SpirvOp.FOrdNotEqual,
                    "Ge" => SpirvOp.FOrdGreaterThanEqual,
                    "Neq" => SpirvOp.FUnordNotEqual,
                    "Nlt" => SpirvOp.FUnordGreaterThanEqual,
                    "Nle" => SpirvOp.FUnordGreaterThan,
                    "Ngt" => SpirvOp.FUnordLessThanEqual,
                    "Nge" => SpirvOp.FUnordLessThan,
                    "Nlg" => SpirvOp.FUnordEqual,
                    _ => SpirvOp.Nop,
                };
                if (operation == SpirvOp.Nop)
                {
                    error = $"unsupported float compare {opcode}";
                    return false;
                }

                condition = _module.AddInstruction(operation, _boolType, left, right);
            }
            else
            {
                var is16 = opcode.EndsWith("I16", StringComparison.Ordinal) ||
                           opcode.EndsWith("U16", StringComparison.Ordinal);
                var signed = opcode.EndsWith("I32", StringComparison.Ordinal) ||
                             opcode.EndsWith("I16", StringComparison.Ordinal);
                var left = is16 && instruction.Control is Gen5Vop3Control halfControl
                    ? EmitVop3HalfBits(instruction, halfControl, 0)
                    : GetRawSource(instruction, 0);
                var right = is16 && instruction.Control is Gen5Vop3Control rightHalfControl
                    ? EmitVop3HalfBits(instruction, rightHalfControl, 1)
                    : GetRawSource(instruction, 1);
                if (is16)
                {
                    left = BitwiseAnd(left, UInt(0xFFFF));
                    right = BitwiseAnd(right, UInt(0xFFFF));
                }
                if (signed)
                {
                    if (is16)
                    {
                        left = ShiftRightArithmetic(ShiftLeftLogical(left, UInt(16)), UInt(16));
                        right = ShiftRightArithmetic(ShiftLeftLogical(right, UInt(16)), UInt(16));
                    }
                    left = Bitcast(_intType, left);
                    right = Bitcast(_intType, right);
                }

                var operation = TrimCompareOpcode(opcode) switch
                {
                    "Eq" => SpirvOp.IEqual,
                    "Ne" => SpirvOp.INotEqual,
                    "Lt" => signed ? SpirvOp.SLessThan : SpirvOp.ULessThan,
                    "Le" => signed ? SpirvOp.SLessThanEqual : SpirvOp.ULessThanEqual,
                    "Gt" => signed ? SpirvOp.SGreaterThan : SpirvOp.UGreaterThan,
                    "Ge" => signed ? SpirvOp.SGreaterThanEqual : SpirvOp.UGreaterThanEqual,
                    _ => SpirvOp.Nop,
                };
                if (operation == SpirvOp.Nop)
                {
                    error = $"unsupported integer compare {opcode}";
                    return false;
                }

                condition = _module.AddInstruction(operation, _boolType, left, right);
            }

            if (_state.Program.Address == 0x0000000500781200ul &&
                ((instruction.Pc == 0x4D4 &&
                  Environment.GetEnvironmentVariable(
                      "SHARPEMU_FORCE_TITLE_COMPARE_4D4") == "1") ||
                 (instruction.Pc == 0x540 &&
                  Environment.GetEnvironmentVariable(
                      "SHARPEMU_FORCE_TITLE_COMPARE_540") == "1")))
            {
                condition = _module.ConstantBool(true);
            }

            // Vector compares fully overwrite the destination mask, but only
            // lanes enabled by EXEC can pass the test: VCC = EXEC & condition.
            // Balloting the raw condition leaks results from disabled lanes
            // into later saveexec/branch sequences.
            var activeCondition = _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                Load(_boolType, _exec),
                condition);
            if (instruction.Control is Gen5DppControl compareDpp)
            {
                activeCondition = _module.AddInstruction(
                    SpirvOp.Select,
                    _boolType,
                    IsDppWriteEnabled(compareDpp),
                    activeCondition,
                    Load(_boolType, _vcc));
            }

            if (opcode.StartsWith("VCmpx", StringComparison.Ordinal))
            {
                // GFX10 VCMPX is EXEC-only. The SDWA bits that older
                // generations exposed as an explicit scalar destination must
                // not overwrite VCC/an SGPR pair here. Guest shaders rely on
                // those registers retaining a saved EXEC mask for later
                // reconvergence.
                StoreWaveMask(126, activeCondition);
            }
            else
            {
                var compareDestination = instruction.Control switch
                {
                    Gen5SdwaControl { ScalarDestination: { } scalarDestination } =>
                        scalarDestination,
                    Gen5Vop3Control { ScalarDestination: { } scalarDestination } =>
                        scalarDestination,
                    _ => 106u,
                };
                StoreWaveMask(compareDestination, activeCondition);
            }

            return true;
        }

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

                var current = LoadS(destination);
                var value = instruction.Opcode switch
                {
                    "SMovkI32" => UInt(immediate),
                    "SAddkI32" => IAdd(current, UInt(immediate)),
                    "SMulkI32" => _module.AddInstruction(
                        SpirvOp.IMul,
                        _uintType,
                        current,
                        UInt(immediate)),
                    _ => 0u,
                };
                if (value == 0)
                {
                    error = $"unsupported scalar immediate {instruction.Opcode}";
                    return false;
                }

                StoreS(destination, value);
                return true;
            }

            if (instruction.Opcode == "SGetpcB64")
            {
                var pc = _state.Program.Address +
                    instruction.Pc +
                    (ulong)(instruction.Words.Count * sizeof(uint));
                StoreS(destination, UInt((uint)pc));
                StoreS(destination + 1, UInt((uint)(pc >> 32)));
                return true;
            }

            if (instruction.Opcode is
                "SMovrelsB32" or "SMovrelsB64" or
                "SMovreldB32" or "SMovreldB64")
            {
                return TryEmitScalarRelativeMove(instruction, destination, out error);
            }

            if (instruction.Opcode.EndsWith("B64", StringComparison.Ordinal) ||
                instruction.Opcode is "SWqmB64" or "SBfeU64" or "SBfeI64" or "SAshrI64")
            {
                return TryEmitScalar64(instruction, destination, out error);
            }

            var left = GetRawSource(instruction, 0);
            if (instruction.Opcode.EndsWith("SaveexecB32", StringComparison.Ordinal))
            {
                var oldExec64 = BooleanToWaveMask(Load(_boolType, _exec));
                var oldExec = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _uintType,
                    oldExec64);
                var notLeft = _module.AddInstruction(SpirvOp.Not, _uintType, left);
                var notOldExec = _module.AddInstruction(SpirvOp.Not, _uintType, oldExec);
                var newExec = instruction.Opcode switch
                {
                    "SAndSaveexecB32" => BitwiseAnd(oldExec, left),
                    "SOrSaveexecB32" => BitwiseOr(oldExec, left),
                    "SXorSaveexecB32" => _module.AddInstruction(
                        SpirvOp.BitwiseXor, _uintType, oldExec, left),
                    "SAndn1SaveexecB32" => BitwiseAnd(notLeft, oldExec),
                    "SAndn2SaveexecB32" => BitwiseAnd(left, notOldExec),
                    "SOrn1SaveexecB32" => BitwiseOr(notLeft, oldExec),
                    "SOrn2SaveexecB32" => BitwiseOr(left, notOldExec),
                    "SNandSaveexecB32" => _module.AddInstruction(
                        SpirvOp.Not, _uintType, BitwiseAnd(left, oldExec)),
                    "SNorSaveexecB32" => _module.AddInstruction(
                        SpirvOp.Not, _uintType, BitwiseOr(left, oldExec)),
                    "SXnorSaveexecB32" => _module.AddInstruction(
                        SpirvOp.Not,
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseXor,
                            _uintType,
                            left,
                            oldExec)),
                    _ => 0u,
                };
                if (newExec == 0)
                {
                    error = $"unsupported scalar 32-bit saveexec opcode {instruction.Opcode}";
                    return false;
                }

                StoreS(destination, oldExec);
                // B32 saveexec is the Wave32 form; EXEC_HI is zero.
                var newExec64 = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    newExec);
                StoreS64(126, newExec64);
                Store(_scc, IsNotZero(newExec));
                return true;
            }

            uint result;
            switch (instruction.Opcode)
            {
                case "SMovB32":
                    result = left;
                    break;
                case "SNotB32":
                    result = _module.AddInstruction(SpirvOp.Not, _uintType, left);
                    StoreS(destination, result);
                    Store(_scc, IsNotZero(result));
                    return true;
                case "SBrevB32":
                    result = _module.AddInstruction(SpirvOp.BitReverse, _uintType, left);
                    StoreS(destination, result);
                    Store(_scc, IsNotZero(result));
                    return true;
                case "SBcnt1I32B32":
                    result = _module.AddInstruction(SpirvOp.BitCount, _uintType, left);
                    StoreS(destination, result);
                    Store(_scc, IsNotZero(result));
                    return true;
                case "SFF1I32B32":
                    result = Ext(73, _uintType, left);
                    StoreS(destination, result);
                    Store(_scc, IsNotZero(result));
                    return true;
                case "SBitset1B32":
                    result = _module.AddInstruction(
                        SpirvOp.BitFieldInsert,
                        _uintType,
                        LoadS(destination),
                        UInt(1),
                        BitwiseAnd(left, UInt(31)),
                        UInt(1));
                    StoreS(destination, result);
                    return true;
                default:
                {
                    if (instruction.Sources.Count < 2)
                    {
                        error = $"missing scalar source for {instruction.Opcode}";
                        return false;
                    }

                    var right = GetRawSource(instruction, 1);
                    switch (instruction.Opcode)
                    {
                        case "SAddU32":
                            result = IAdd(left, right);
                            Store(_scc, _module.AddInstruction(
                                SpirvOp.ULessThan,
                                _boolType,
                                result,
                                left));
                            break;
                        case "SSubU32":
                            result = _module.AddInstruction(
                                SpirvOp.ISub,
                                _uintType,
                                left,
                                right);
                            Store(_scc, _module.AddInstruction(
                                SpirvOp.UGreaterThan,
                                _boolType,
                                right,
                                left));
                            break;
                        case "SAddI32":
                            result = IAdd(left, right);
                            Store(_scc, SignedAddOverflow(left, right, result));
                            break;
                        case "SSubI32":
                            result = _module.AddInstruction(
                                SpirvOp.ISub,
                                _uintType,
                                left,
                                right);
                            Store(_scc, SignedSubOverflow(left, right, result));
                            break;
                        case "SAddcU32":
                        {
                            var carryIn = _module.AddInstruction(
                                SpirvOp.Select,
                                _uintType,
                                Load(_boolType, _scc),
                                UInt(1),
                                UInt(0));
                            var partial = IAdd(left, right);
                            result = IAdd(partial, carryIn);
                            var firstCarry = _module.AddInstruction(
                                SpirvOp.ULessThan,
                                _boolType,
                                partial,
                                left);
                            var secondCarry = _module.AddInstruction(
                                SpirvOp.ULessThan,
                                _boolType,
                                result,
                                partial);
                            Store(
                                _scc,
                                _module.AddInstruction(
                                    SpirvOp.LogicalOr,
                                    _boolType,
                                    firstCarry,
                                    secondCarry));
                            break;
                        }
                        case "SSubbU32":
                        {
                            var borrow = _module.AddInstruction(
                                SpirvOp.Select,
                                _uintType,
                                Load(_boolType, _scc),
                                UInt(1),
                                UInt(0));
                            var partial = _module.AddInstruction(
                                SpirvOp.ISub,
                                _uintType,
                                left,
                                right);
                            result = _module.AddInstruction(
                                SpirvOp.ISub,
                                _uintType,
                                partial,
                                borrow);
                            var firstBorrow = _module.AddInstruction(
                                SpirvOp.UGreaterThan,
                                _boolType,
                                right,
                                left);
                            var secondBorrow = _module.AddInstruction(
                                SpirvOp.LogicalAnd,
                                _boolType,
                                _module.AddInstruction(
                                    SpirvOp.IEqual,
                                    _boolType,
                                    borrow,
                                    UInt(1)),
                                _module.AddInstruction(
                                    SpirvOp.IEqual,
                                    _boolType,
                                    right,
                                    left));
                            Store(
                                _scc,
                                _module.AddInstruction(
                                    SpirvOp.LogicalOr,
                                    _boolType,
                                    firstBorrow,
                                    secondBorrow));
                            break;
                        }
                        case "SMulI32":
                            result = _module.AddInstruction(
                                SpirvOp.IMul,
                                _uintType,
                                left,
                                right);
                            break;
                        case "SMulHiU32":
                        {
                            var product = _module.AddInstruction(
                                SpirvOp.IMul,
                                _ulongType,
                                _module.AddInstruction(SpirvOp.UConvert, _ulongType, left),
                                _module.AddInstruction(SpirvOp.UConvert, _ulongType, right));
                            result = _module.AddInstruction(
                                SpirvOp.UConvert,
                                _uintType,
                                ShiftRightLogical64(
                                    product,
                                    _module.Constant64(_ulongType, 32)));
                            break;
                        }
                        case "SMulHiI32":
                        {
                            var product = _module.AddInstruction(
                                SpirvOp.IMul,
                                _longType,
                                _module.AddInstruction(
                                    SpirvOp.SConvert,
                                    _longType,
                                    Bitcast(_intType, left)),
                                _module.AddInstruction(
                                    SpirvOp.SConvert,
                                    _longType,
                                    Bitcast(_intType, right)));
                            result = Bitcast(
                                _uintType,
                                _module.AddInstruction(
                                    SpirvOp.SConvert,
                                    _intType,
                                    _module.AddInstruction(
                                        SpirvOp.ShiftRightArithmetic,
                                        _longType,
                                        product,
                                        _module.Constant64(_longType, 32))));
                            break;
                        }
                        case "SAndB32":
                            result = BitwiseAnd(left, right);
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SOrB32":
                            result = _module.AddInstruction(
                                SpirvOp.BitwiseOr,
                                _uintType,
                                left,
                                right);
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SXorB32":
                            result = _module.AddInstruction(
                                SpirvOp.BitwiseXor,
                                _uintType,
                                left,
                                right);
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SAndn2B32":
                            result = BitwiseAnd(
                                left,
                                _module.AddInstruction(SpirvOp.Not, _uintType, right));
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SOrn2B32":
                            result = _module.AddInstruction(
                                SpirvOp.BitwiseOr,
                                _uintType,
                                left,
                                _module.AddInstruction(SpirvOp.Not, _uintType, right));
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SNandB32":
                            result = _module.AddInstruction(
                                SpirvOp.Not,
                                _uintType,
                                BitwiseAnd(left, right));
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SNorB32":
                            result = _module.AddInstruction(
                                SpirvOp.Not,
                                _uintType,
                                _module.AddInstruction(
                                    SpirvOp.BitwiseOr,
                                    _uintType,
                                    left,
                                    right));
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SXnorB32":
                            result = _module.AddInstruction(
                                SpirvOp.Not,
                                _uintType,
                                _module.AddInstruction(
                                    SpirvOp.BitwiseXor,
                                    _uintType,
                                    left,
                                    right));
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SLshlB32":
                            result = ShiftLeftLogical(left, right);
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SLshrB32":
                            result = ShiftRightLogical(
                                left,
                                BitwiseAnd(right, UInt(31)));
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SAshrI32":
                            result = ShiftRightArithmetic(left, right);
                            Store(_scc, IsNotZero(result));
                            break;
                        case "SAbsdiffI32":
                        {
                            var difference = _module.AddInstruction(
                                SpirvOp.ISub,
                                _uintType,
                                left,
                                right);
                            var negative = IsNotZero(
                                BitwiseAnd(difference, UInt(0x8000_0000)));
                            result = _module.AddInstruction(
                                SpirvOp.Select,
                                _uintType,
                                negative,
                                _module.AddInstruction(
                                    SpirvOp.ISub,
                                    _uintType,
                                    UInt(0),
                                    difference),
                                difference);
                            Store(_scc, IsNotZero(result));
                            break;
                        }
                        case "SBfmB32":
                            result = _module.AddInstruction(
                                SpirvOp.BitFieldInsert,
                                _uintType,
                                UInt(0),
                                UInt(uint.MaxValue),
                                BitwiseAnd(right, UInt(31)),
                                BitwiseAnd(left, UInt(31)));
                            break;
                        case "SBfeU32":
                        case "SBfeI32":
                        {
                            var offset = BitwiseAnd(right, UInt(31));
                            var requestedWidth = BitwiseAnd(
                                ShiftRightLogical(right, UInt(16)),
                                UInt(0x7F));
                            var remaining = _module.AddInstruction(
                                SpirvOp.ISub,
                                _uintType,
                                UInt(32),
                                offset);
                            var width = Ext(
                                38,
                                _uintType,
                                requestedWidth,
                                remaining);
                            result = instruction.Opcode == "SBfeI32"
                                ? Bitcast(
                                    _uintType,
                                    _module.AddInstruction(
                                        SpirvOp.BitFieldSExtract,
                                        _intType,
                                        Bitcast(_intType, left),
                                        offset,
                                        width))
                                : _module.AddInstruction(
                                    SpirvOp.BitFieldUExtract,
                                    _uintType,
                                    left,
                                    offset,
                                    width);
                            Store(_scc, IsNotZero(result));
                            break;
                        }
                        case "SCselectB32":
                            result = _module.AddInstruction(
                                SpirvOp.Select,
                                _uintType,
                                Load(_boolType, _scc),
                                left,
                                right);
                            break;
                        case "SMinU32":
                            result = Ext(38, _uintType, left, right);
                            Store(
                                _scc,
                                _module.AddInstruction(
                                    SpirvOp.ULessThan,
                                    _boolType,
                                    left,
                                    right));
                            break;
                        case "SMinI32":
                            result = Bitcast(
                                _uintType,
                                Ext(39, _intType, Bitcast(_intType, left), Bitcast(_intType, right)));
                            Store(
                                _scc,
                                _module.AddInstruction(
                                    SpirvOp.SLessThan,
                                    _boolType,
                                    Bitcast(_intType, left),
                                    Bitcast(_intType, right)));
                            break;
                        case "SMaxU32":
                            result = Ext(41, _uintType, left, right);
                            Store(
                                _scc,
                                _module.AddInstruction(
                                    SpirvOp.UGreaterThan,
                                    _boolType,
                                    left,
                                    right));
                            break;
                        case "SMaxI32":
                            result = Bitcast(
                                _uintType,
                                Ext(42, _intType, Bitcast(_intType, left), Bitcast(_intType, right)));
                            Store(
                                _scc,
                                _module.AddInstruction(
                                    SpirvOp.SGreaterThan,
                                    _boolType,
                                    Bitcast(_intType, left),
                                    Bitcast(_intType, right)));
                            break;
                        case "SLshl1AddU32":
                        case "SLshl2AddU32":
                        case "SLshl3AddU32":
                        case "SLshl4AddU32":
                        {
                            var shift = (uint)(instruction.Opcode[5] - '0');
                            result = IAdd(
                                ShiftLeftLogical(left, UInt(shift)),
                                right);
                            break;
                        }
                        case "SPackLlB32B16":
                            result = BitwiseOr(
                                BitwiseAnd(left, UInt(0xFFFF)),
                                ShiftLeftLogical(right, UInt(16)));
                            break;
                        case "SPackLhB32B16":
                            result = BitwiseOr(
                                BitwiseAnd(left, UInt(0xFFFF)),
                                BitwiseAnd(right, UInt(0xFFFF0000)));
                            break;
                        case "SPackHhB32B16":
                            result = BitwiseOr(
                                ShiftRightLogical(left, UInt(16)),
                                BitwiseAnd(right, UInt(0xFFFF0000)));
                            break;
                        default:
                            error = $"unsupported scalar opcode {instruction.Opcode}";
                            return false;
                    }

                    break;
                }
            }

            StoreS(destination, result);
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

            var left = GetRawSource(instruction, 0);
            var right = GetRawSource(instruction, 1);
            if (instruction.Opcode is "SBitcmp0B32" or "SBitcmp1B32")
            {
                var shifted = ShiftRightLogical(
                    left,
                    BitwiseAnd(right, UInt(31)));
                var isSet = IsNotZero(BitwiseAnd(shifted, UInt(1)));
                Store(
                    _scc,
                    instruction.Opcode == "SBitcmp1B32"
                        ? isSet
                        : _module.AddInstruction(
                            SpirvOp.LogicalNot,
                            _boolType,
                            isSet));
                return true;
            }

            if (instruction.Opcode is "SBitcmp0B64" or "SBitcmp1B64")
            {
                var wideLeft = GetRawSource64(instruction, 0);
                var wideIndex = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    right);
                var shifted = ShiftRightLogical64(wideLeft, wideIndex);
                var isSet = IsNotZero64(
                    BitwiseAnd64(
                        shifted,
                        _module.Constant64(_ulongType, 1)));
                Store(
                    _scc,
                    instruction.Opcode == "SBitcmp1B64"
                        ? isSet
                        : _module.AddInstruction(
                            SpirvOp.LogicalNot,
                            _boolType,
                            isSet));
                return true;
            }

            var operation = instruction.Opcode switch
            {
                "SCmpEqI32" or "SCmpEqU32" => SpirvOp.IEqual,
                "SCmpLgI32" or "SCmpLgU32" => SpirvOp.INotEqual,
                "SCmpGtI32" => SpirvOp.SGreaterThan,
                "SCmpGeI32" => SpirvOp.SGreaterThanEqual,
                "SCmpLtI32" => SpirvOp.SLessThan,
                "SCmpLeI32" => SpirvOp.SLessThanEqual,
                "SCmpGtU32" => SpirvOp.UGreaterThan,
                "SCmpGeU32" => SpirvOp.UGreaterThanEqual,
                "SCmpLtU32" => SpirvOp.ULessThan,
                "SCmpLeU32" => SpirvOp.ULessThanEqual,
                _ => SpirvOp.Nop,
            };
            if (operation == SpirvOp.Nop)
            {
                error = $"unsupported scalar compare {instruction.Opcode}";
                return false;
            }

            if (instruction.Opcode.EndsWith("I32", StringComparison.Ordinal))
            {
                left = Bitcast(_intType, left);
                right = Bitcast(_intType, right);
            }

            Store(_scc, _module.AddInstruction(operation, _boolType, left, right));
            return true;
        }

        // Metal does not expose double precision and Vulkan devices may omit
        // shaderFloat64, so F64 compares are evaluated from their IEEE-754 bit
        // representation. The sign-aware sortable key gives the same ordering
        // for every non-NaN value while explicitly folding +0 and -0 together.
        private uint EmitFloat64Compare(Gen5ShaderInstruction instruction)
        {
            var left = GetFloat64SourceBits(instruction, 0);
            var right = GetFloat64SourceBits(instruction, 1);
            var sign = _module.Constant64(_ulongType, 0x8000_0000_0000_0000UL);
            var magnitude = _module.Constant64(_ulongType, 0x7FFF_FFFF_FFFF_FFFFUL);
            var exponent = _module.Constant64(_ulongType, 0x7FF0_0000_0000_0000UL);
            var mantissa = _module.Constant64(_ulongType, 0x000F_FFFF_FFFF_FFFFUL);
            var zero64 = _module.Constant64(_ulongType, 0);

            uint Logical(SpirvOp operation, uint a, uint b) =>
                _module.AddInstruction(operation, _boolType, a, b);
            uint Not(uint value) => _module.AddInstruction(SpirvOp.LogicalNot, _boolType, value);
            uint Equal64(uint a, uint b) => _module.AddInstruction(SpirvOp.IEqual, _boolType, a, b);
            uint IsNan(uint value)
            {
                var exponentIsOnes = Equal64(BitwiseAnd64(value, exponent), exponent);
                var hasMantissa = IsNotZero64(BitwiseAnd64(value, mantissa));
                return Logical(SpirvOp.LogicalAnd, exponentIsOnes, hasMantissa);
            }

            uint SortKey(uint value)
            {
                var negative = IsNotZero64(BitwiseAnd64(value, sign));
                var inverted = _module.AddInstruction(SpirvOp.Not, _ulongType, value);
                var positive = _module.AddInstruction(SpirvOp.BitwiseXor, _ulongType, value, sign);
                return _module.AddInstruction(SpirvOp.Select, _ulongType, negative, inverted, positive);
            }

            var unordered = Logical(SpirvOp.LogicalOr, IsNan(left), IsNan(right));
            var ordered = Not(unordered);
            var bothZero = Logical(
                SpirvOp.LogicalAnd,
                Equal64(BitwiseAnd64(left, magnitude), zero64),
                Equal64(BitwiseAnd64(right, magnitude), zero64));
            var equalValue = Logical(SpirvOp.LogicalOr, Equal64(left, right), bothZero);
            var equal = Logical(SpirvOp.LogicalAnd, ordered, equalValue);
            var less = Logical(
                SpirvOp.LogicalAnd,
                ordered,
                _module.AddInstruction(
                    SpirvOp.ULessThan,
                    _boolType,
                    SortKey(left),
                    SortKey(right)));
            var greater = Logical(
                SpirvOp.LogicalAnd,
                ordered,
                _module.AddInstruction(
                    SpirvOp.UGreaterThan,
                    _boolType,
                    SortKey(left),
                    SortKey(right)));
            var lessEqual = Logical(SpirvOp.LogicalOr, less, equal);
            var greaterEqual = Logical(SpirvOp.LogicalOr, greater, equal);
            var lessGreater = Logical(SpirvOp.LogicalOr, less, greater);

            return TrimCompareOpcode(instruction.Opcode) switch
            {
                "F" => _module.ConstantBool(false),
                "Lt" => less,
                "Eq" => equal,
                "Le" => lessEqual,
                "Gt" => greater,
                "Lg" => lessGreater,
                "Ge" => greaterEqual,
                "O" => ordered,
                "U" => unordered,
                "Nge" => Not(greaterEqual),
                "Nlg" => Not(lessGreater),
                "Ngt" => Not(greater),
                "Nle" => Not(lessEqual),
                "Neq" => Not(equal),
                "Nlt" => Not(less),
                "Tru" => _module.ConstantBool(true),
                _ => _module.ConstantBool(false),
            };
        }

        private uint GetFloat64SourceBits(
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
                ? _module.Constant64(
                    _ulongType,
                    BitConverter.DoubleToUInt64Bits(constant.Value))
                : GetRawSource64(instruction, sourceIndex);
            if (instruction.Control is Gen5Vop3Control control)
            {
                if ((control.AbsoluteMask & (1u << sourceIndex)) != 0)
                {
                    bits = BitwiseAnd64(
                        bits,
                        _module.Constant64(_ulongType, 0x7FFF_FFFF_FFFF_FFFFUL));
                }

                if ((control.NegateMask & (1u << sourceIndex)) != 0)
                {
                    bits = _module.AddInstruction(
                        SpirvOp.BitwiseXor,
                        _ulongType,
                        bits,
                        _module.Constant64(_ulongType, 0x8000_0000_0000_0000UL));
                }
            }

            return bits;
        }

        private static string TrimCompareOpcode(string opcode)
        {
            var trimmed = opcode.StartsWith("VCmpx", StringComparison.Ordinal)
                ? opcode["VCmpx".Length..]
                : opcode["VCmp".Length..];
            return trimmed[..^3];
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

            var m0 = LoadS(124);
            var relativeSource = instruction.Opcode.StartsWith(
                "SMovrels",
                StringComparison.Ordinal);
            var is64 = instruction.Opcode.EndsWith("B64", StringComparison.Ordinal);
            var source = instruction.Sources[0].Value;

            if (!is64)
            {
                var value = relativeSource
                    ? LoadSRelative(source, m0)
                    : LoadS(source);
                if (relativeSource)
                {
                    StoreS(destination, value);
                }
                else
                {
                    StoreSRelative(destination, m0, value);
                }

                return true;
            }

            var low = relativeSource
                ? LoadSRelative(source, m0)
                : LoadS(source);
            var high = relativeSource
                ? LoadSRelative(source + 1, m0)
                : LoadS(source + 1);
            if (relativeSource)
            {
                StoreS(destination, low);
                StoreS(destination + 1, high);
            }
            else
            {
                StoreSRelative(destination, m0, low);
                StoreSRelative(destination + 1, m0, high);
            }

            return true;
        }

        private bool TryEmitScalarCompareK(
            Gen5ShaderInstruction instruction,
            uint destination,
            uint immediate,
            out string error)
        {
            error = string.Empty;
            var left = LoadS(destination);
            var right = UInt(immediate);
            var operation = instruction.Opcode switch
            {
                "SCmpkEqI32" or "SCmpkEqU32" => SpirvOp.IEqual,
                "SCmpkLgI32" or "SCmpkLgU32" => SpirvOp.INotEqual,
                "SCmpkGtI32" => SpirvOp.SGreaterThan,
                "SCmpkGeI32" => SpirvOp.SGreaterThanEqual,
                "SCmpkLtI32" => SpirvOp.SLessThan,
                "SCmpkLeI32" => SpirvOp.SLessThanEqual,
                "SCmpkGtU32" => SpirvOp.UGreaterThan,
                "SCmpkGeU32" => SpirvOp.UGreaterThanEqual,
                "SCmpkLtU32" => SpirvOp.ULessThan,
                "SCmpkLeU32" => SpirvOp.ULessThanEqual,
                _ => SpirvOp.Nop,
            };
            if (operation == SpirvOp.Nop)
            {
                error = $"unsupported scalar immediate compare {instruction.Opcode}";
                return false;
            }

            if (instruction.Opcode.EndsWith("I32", StringComparison.Ordinal))
            {
                left = Bitcast(_intType, left);
                right = Bitcast(_intType, right);
            }

            Store(_scc, _module.AddInstruction(operation, _boolType, left, right));
            return true;
        }

        private bool TryEmitScalar64(
            Gen5ShaderInstruction instruction,
            uint destination,
            out string error)
        {
            error = string.Empty;
            var left = GetRawSource64(instruction, 0);
            if (instruction.Opcode.EndsWith("SaveexecB64", StringComparison.Ordinal))
            {
                var oldExec = BooleanToWaveMask(Load(_boolType, _exec));
                var notLeft = _module.AddInstruction(SpirvOp.Not, _ulongType, left);
                var newExec = instruction.Opcode switch
                {
                    "SAndSaveexecB64" => _module.AddInstruction(
                        SpirvOp.BitwiseAnd, _ulongType, oldExec, left),
                    "SOrSaveexecB64" => _module.AddInstruction(
                        SpirvOp.BitwiseOr, _ulongType, oldExec, left),
                    "SXorSaveexecB64" => _module.AddInstruction(
                        SpirvOp.BitwiseXor, _ulongType, oldExec, left),
                    "SAndn2SaveexecB64" => _module.AddInstruction(
                        SpirvOp.BitwiseAnd,
                        _ulongType,
                        left,
                        _module.AddInstruction(
                            SpirvOp.Not,
                            _ulongType,
                            oldExec)),
                    "SAndn1SaveexecB64" => _module.AddInstruction(
                        SpirvOp.BitwiseAnd,
                        _ulongType,
                        notLeft,
                        oldExec),
                    "SOrn1SaveexecB64" => _module.AddInstruction(
                        SpirvOp.BitwiseOr,
                        _ulongType,
                        notLeft,
                        oldExec),
                    "SOrn2SaveexecB64" => _module.AddInstruction(
                        SpirvOp.BitwiseOr,
                        _ulongType,
                        left,
                        _module.AddInstruction(
                            SpirvOp.Not,
                            _ulongType,
                            oldExec)),
                    "SNandSaveexecB64" => _module.AddInstruction(
                        SpirvOp.Not,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseAnd,
                            _ulongType,
                            left,
                            oldExec)),
                    "SNorSaveexecB64" => _module.AddInstruction(
                        SpirvOp.Not,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseOr,
                            _ulongType,
                            left,
                            oldExec)),
                    "SXnorSaveexecB64" => _module.AddInstruction(
                        SpirvOp.Not,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseXor,
                            _ulongType,
                            left,
                            oldExec)),
                    _ => 0u,
                };
                if (newExec == 0)
                {
                    error =
                        $"unsupported scalar 64-bit opcode {instruction.Opcode}";
                    return false;
                }

                StoreS64(destination, oldExec);
                StoreS64(126, newExec);
                Store(_scc, IsNotZero64(newExec));
                return true;
            }

            if (instruction.Opcode is "SLshlB64" or "SLshrB64" or "SAshrI64")
            {
                if (instruction.Sources.Count < 2)
                {
                    error = "missing scalar 64-bit shift source";
                    return false;
                }

                var shift = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    GetRawSource(instruction, 1));
                var shiftedValue = instruction.Opcode switch
                {
                    "SLshlB64" => ShiftLeftLogical64(left, shift),
                    "SLshrB64" => ShiftRightLogical64(left, shift),
                    _ => Bitcast(
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.ShiftRightArithmetic,
                            _longType,
                            Bitcast(_longType, left),
                            Bitcast(
                                _longType,
                                BitwiseAnd64(
                                    shift,
                                    _module.Constant64(_ulongType, 63))))),
                };
                StoreS64(destination, shiftedValue);
                Store(_scc, IsNotZero64(shiftedValue));
                return true;
            }

            if (instruction.Opcode is "SBfeU64" or "SBfeI64")
            {
                if (instruction.Sources.Count < 2)
                {
                    error = "missing scalar 64-bit bitfield source";
                    return false;
                }

                var control = GetRawSource(instruction, 1);
                var offset = BitwiseAnd(control, UInt(63));
                var requestedWidth = BitwiseAnd(
                    ShiftRightLogical(control, UInt(16)),
                    UInt(0x7F));
                var remaining = _module.AddInstruction(
                    SpirvOp.ISub,
                    _uintType,
                    UInt(64),
                    offset);
                var width = Ext(
                    38,
                    _uintType,
                    requestedWidth,
                    remaining);
                var offset64 = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    offset);
                var width64 = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    width);
                var one64 = _module.Constant64(_ulongType, 1);
                var shifted = ShiftRightLogical64(left, offset64);
                var partialMask = _module.AddInstruction(
                    SpirvOp.ISub,
                    _ulongType,
                    ShiftLeftLogical64(one64, width64),
                    one64);
                var fullWidth = _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    width,
                    UInt(64));
                var mask = _module.AddInstruction(
                    SpirvOp.Select,
                    _ulongType,
                    fullWidth,
                    _module.Constant64(_ulongType, ulong.MaxValue),
                    partialMask);
                var extracted = _module.AddInstruction(
                    SpirvOp.BitwiseAnd,
                    _ulongType,
                    shifted,
                    mask);
                if (instruction.Opcode == "SBfeI64")
                {
                    var signShift = _module.AddInstruction(
                        SpirvOp.ISub,
                        _uintType,
                        width,
                        UInt(1));
                    var signBit = ShiftLeftLogical64(
                        one64,
                        _module.AddInstruction(
                            SpirvOp.UConvert,
                            _ulongType,
                            signShift));
                    var signExtended = _module.AddInstruction(
                        SpirvOp.ISub,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseXor,
                            _ulongType,
                            extracted,
                            signBit),
                        signBit);
                    extracted = _module.AddInstruction(
                        SpirvOp.Select,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.IEqual,
                            _boolType,
                            width,
                            UInt(0)),
                        _module.Constant64(_ulongType, 0),
                        signExtended);
                }

                StoreS64(destination, extracted);
                Store(_scc, IsNotZero64(extracted));
                return true;
            }

            if (instruction.Opcode == "SBfmB64")
            {
                if (instruction.Sources.Count < 2)
                {
                    error = "missing scalar 64-bit bitfield-mask source";
                    return false;
                }

                var width = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    BitwiseAnd(GetRawSource(instruction, 0), UInt(63)));
                var offset = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    BitwiseAnd(GetRawSource(instruction, 1), UInt(63)));
                // Width is masked to 0..63, so (1 << width) never invokes an
                // undefined 64-bit shift. This naturally yields zero for a
                // zero-width mask and avoids OpBitFieldInsert, which the
                // MoltenVK/SPIRV-Cross path rejects for 64-bit integers.
                var lowMask = _module.AddInstruction(
                    SpirvOp.ISub,
                    _ulongType,
                    ShiftLeftLogical64(
                        _module.Constant64(_ulongType, 1),
                        width),
                    _module.Constant64(_ulongType, 1));
                var maskValue = ShiftLeftLogical64(lowMask, offset);
                StoreS64(destination, maskValue);
                Store(_scc, IsNotZero64(maskValue));
                return true;
            }

            uint value;
            if (instruction.Opcode == "SMovB64")
            {
                value = left;
            }
            else if (instruction.Opcode == "SWqmB64")
            {
                var quadAny = _module.AddInstruction(
                    SpirvOp.BitwiseOr,
                    _ulongType,
                    left,
                    _module.AddInstruction(
                        SpirvOp.BitwiseOr,
                        _ulongType,
                        ShiftRightLogical64(left, _module.Constant64(_ulongType, 1)),
                        _module.AddInstruction(
                            SpirvOp.BitwiseOr,
                            _ulongType,
                            ShiftRightLogical64(left, _module.Constant64(_ulongType, 2)),
                            ShiftRightLogical64(left, _module.Constant64(_ulongType, 3)))));
                quadAny = _module.AddInstruction(
                    SpirvOp.BitwiseAnd,
                    _ulongType,
                    quadAny,
                    _module.Constant64(_ulongType, 0x1111_1111_1111_1111UL));
                value = _module.AddInstruction(
                    SpirvOp.IMul,
                    _ulongType,
                    quadAny,
                    _module.Constant64(_ulongType, 0xFUL));
            }
            else if (instruction.Opcode == "SNotB64")
            {
                value = _module.AddInstruction(SpirvOp.Not, _ulongType, left);
            }
            else
            {
                if (instruction.Sources.Count < 2)
                {
                    error = "missing scalar 64-bit source";
                    return false;
                }

                var right = GetRawSource64(instruction, 1);
                value = instruction.Opcode switch
                {
                    "SAndB64" => _module.AddInstruction(
                        SpirvOp.BitwiseAnd, _ulongType, left, right),
                    "SOrB64" => _module.AddInstruction(
                        SpirvOp.BitwiseOr, _ulongType, left, right),
                    "SXorB64" => _module.AddInstruction(
                        SpirvOp.BitwiseXor, _ulongType, left, right),
                    "SNandB64" => _module.AddInstruction(
                        SpirvOp.Not,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseAnd, _ulongType, left, right)),
                    "SNorB64" => _module.AddInstruction(
                        SpirvOp.Not,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseOr, _ulongType, left, right)),
                    "SXnorB64" => _module.AddInstruction(
                        SpirvOp.Not,
                        _ulongType,
                        _module.AddInstruction(
                            SpirvOp.BitwiseXor, _ulongType, left, right)),
                    "SAndn1B64" => _module.AddInstruction(
                        SpirvOp.BitwiseAnd,
                        _ulongType,
                        _module.AddInstruction(SpirvOp.Not, _ulongType, left),
                        right),
                    "SAndn2B64" => _module.AddInstruction(
                        SpirvOp.BitwiseAnd,
                        _ulongType,
                        left,
                        _module.AddInstruction(SpirvOp.Not, _ulongType, right)),
                    "SOrn1B64" => _module.AddInstruction(
                        SpirvOp.BitwiseOr,
                        _ulongType,
                        _module.AddInstruction(SpirvOp.Not, _ulongType, left),
                        right),
                    "SOrn2B64" => _module.AddInstruction(
                        SpirvOp.BitwiseOr,
                        _ulongType,
                        left,
                        _module.AddInstruction(SpirvOp.Not, _ulongType, right)),
                    "SCselectB64" => _module.AddInstruction(
                        SpirvOp.Select,
                        _ulongType,
                        Load(_boolType, _scc),
                        left,
                        right),
                    _ => 0,
                };
                if (value == 0)
                {
                    error = $"unsupported scalar 64-bit opcode {instruction.Opcode}";
                    return false;
                }
            }

            StoreS64(destination, value);
            if (instruction.Opcode is
                "SNotB64" or
                "SAndB64" or
                "SOrB64" or
                "SXorB64" or
                "SAndn1B64" or
                "SAndn2B64" or
                "SOrn1B64" or
                "SOrn2B64" or
                "SNandB64" or
                "SNorB64" or
                "SXnorB64")
            {
                Store(_scc, IsNotZero64(value));
            }

            return true;
        }

        private uint GetRawSource(
            Gen5ShaderInstruction instruction,
            int sourceIndex,
            bool applySdwaIntegerModifiers = true)
        {
            if ((uint)sourceIndex >= instruction.Sources.Count)
            {
                throw new InvalidOperationException($"missing source {sourceIndex}");
            }

            var operand = instruction.Sources[sourceIndex];
            uint value = operand.Kind switch
            {
                Gen5OperandKind.VectorRegister => LoadV(operand.Value),
                Gen5OperandKind.ScalarRegister => LoadS(operand.Value),
                Gen5OperandKind.LiteralConstant => UInt(operand.Value),
                Gen5OperandKind.EncodedConstant when operand.Value == 251 =>
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        LogicalNot(SubgroupAny(Load(_boolType, _vcc))),
                        UInt(1),
                        UInt(0)),
                Gen5OperandKind.EncodedConstant when operand.Value == 252 =>
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        LogicalNot(SubgroupAny(Load(_boolType, _exec))),
                        UInt(1),
                        UInt(0)),
                Gen5OperandKind.EncodedConstant when operand.Value == 253 =>
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        Load(_boolType, _scc),
                        UInt(1),
                        UInt(0)),
                Gen5OperandKind.EncodedConstant when TryDecodeInlineConstant(
                    operand.Value,
                    out var inline) => UInt(inline),
                _ => throw new InvalidOperationException($"unsupported source {operand}"),
            };

            // DPP16 remaps src0 across lanes before the VALU operation. The IR
            // has always preserved this control, but treating it as an ordinary
            // local VGPR read breaks every wave reduction used by the XPR
            // renderer (min/max/OR scans become value-with-self operations).
            if (sourceIndex == 0 &&
                instruction.Control is Gen5DppControl dpp)
            {
                value = ApplyDppSource(dpp, value);
            }
            else if (sourceIndex == 0 &&
                     instruction.Control is Gen5Dpp8Control dpp8)
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
                    0 => BitwiseAnd(value, UInt(0xFF)),
                    1 => BitwiseAnd(ShiftRightLogical(value, UInt(8)), UInt(0xFF)),
                    2 => BitwiseAnd(ShiftRightLogical(value, UInt(16)), UInt(0xFF)),
                    3 => BitwiseAnd(ShiftRightLogical(value, UInt(24)), UInt(0xFF)),
                    4 => BitwiseAnd(value, UInt(0xFFFF)),
                    5 => BitwiseAnd(ShiftRightLogical(value, UInt(16)), UInt(0xFFFF)),
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
                    value = Bitcast(
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.BitFieldSExtract,
                            _intType,
                            Bitcast(_intType, value),
                            UInt(0),
                            UInt(width)));
                }

                if (applySdwaIntegerModifiers)
                {
                    // SDWA ABS/NEG are floating-point sign-bit modifiers even on
                    // a bit-move opcode: ABS clears the sign bit, NEG flips it.
                    // Two's-complement negating the raw bits instead turns 1.0
                    // into -4.0 and -3.0 into 1.5, which silently skews every
                    // pass that y-flips its clip position with an SDWA-negated
                    // V_MOV_B32 - the whole of UE's DrawRectangle.
                    var signBit = selector switch
                    {
                        <= 3 => 0x80u,
                        4 or 5 => 0x8000u,
                        _ => 0x80000000u,
                    };

                    if ((sdwa.AbsoluteMask & (1u << sourceIndex)) != 0)
                    {
                        value = BitwiseAnd(value, UInt(~signBit));
                    }

                    if ((sdwa.NegateMask & (1u << sourceIndex)) != 0)
                    {
                        value = _module.AddInstruction(
                            SpirvOp.BitwiseXor,
                            _uintType,
                            value,
                            UInt(signBit));
                    }
                }
            }

            return value;
        }

        private uint ApplyDpp8Source(Gen5Dpp8Control control, uint value)
        {
            var lane = GuestWaveLane();
            var laneInGroup = BitwiseAnd(lane, UInt(7));
            var selector = UInt(control.LaneSelectors & 7);
            for (var index = 1u; index < 8; index++)
            {
                selector = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    _module.AddInstruction(
                        SpirvOp.IEqual,
                        _boolType,
                        laneInGroup,
                        UInt(index)),
                    UInt((control.LaneSelectors >> checked((int)(index * 3))) & 7),
                    selector);
            }

            var targetLane = IAdd(BitwiseAnd(lane, UInt(0xFFFF_FFF8)), selector);
            targetLane = BitwiseAnd(targetLane, UInt(31));
            var shuffled = _module.AddInstruction(
                SpirvOp.GroupNonUniformShuffle,
                _uintType,
                UInt(3),
                value,
                targetLane);
            if (control.FetchInactive)
            {
                return shuffled;
            }

            var activeWord = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                Load(_boolType, _exec),
                UInt(1),
                UInt(0));
            var sourceActive = IsNotZero(
                _module.AddInstruction(
                    SpirvOp.GroupNonUniformShuffle,
                    _uintType,
                    UInt(3),
                    activeWord,
                    targetLane));
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                sourceActive,
                shuffled,
                UInt(0));
        }

        private uint ApplyDppSource(Gen5DppControl control, uint value)
        {
            GetDppSourceLane(control, out var targetLane, out var inRange);
            var lane = GuestWaveLane();
            var safeTarget = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                inRange,
                targetLane,
                lane);
            safeTarget = BitwiseAnd(safeTarget, UInt(31));
            var shuffled = _module.AddInstruction(
                SpirvOp.GroupNonUniformShuffle,
                _uintType,
                UInt(3),
                value,
                safeTarget);

            var sourceAvailable = inRange;
            if (!control.FetchInactive)
            {
                var activeWord = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    Load(_boolType, _exec),
                    UInt(1),
                    UInt(0));
                var shuffledActive = _module.AddInstruction(
                    SpirvOp.GroupNonUniformShuffle,
                    _uintType,
                    UInt(3),
                    activeWord,
                    safeTarget);
                sourceAvailable = _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    sourceAvailable,
                    IsNotZero(shuffledActive));
            }

            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                sourceAvailable,
                shuffled,
                UInt(0));
        }

        private uint ApplySdwaDestination(
            Gen5SdwaControl control,
            uint value,
            uint previous)
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
            var fieldMask = lowMask << checked((int)shift);
            var upperStart = shift + width;
            var upperMask = upperStart == 32
                ? 0u
                : uint.MaxValue << checked((int)upperStart);
            var positioned = ShiftLeftLogical(
                BitwiseAnd(value, UInt(lowMask)),
                UInt(shift));
            return control.DestinationUnused switch
            {
                0 => positioned,
                1 => BitwiseOr(
                    positioned,
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        IsNotZero(BitwiseAnd(positioned, UInt(1u << checked((int)(shift + width - 1))))),
                        UInt(upperMask),
                        UInt(0))),
                2 => BitwiseOr(
                    BitwiseAnd(previous, UInt(~fieldMask)),
                    positioned),
                _ => throw new InvalidOperationException("reserved SDWA destination-unused mode"),
            };
        }

        private static bool IsSupportedDppControl(uint control) =>
            control <= 0xFF ||
            control is >= 0x101 and <= 0x10F or
                >= 0x111 and <= 0x11F or
                >= 0x121 and <= 0x12F or
                0x140 or 0x141 or
                >= 0x150 and <= 0x15F or
                >= 0x160 and <= 0x16F;

        private void GetDppSourceLane(
            Gen5DppControl control,
            out uint targetLane,
            out uint inRange)
        {
            var lane = GuestWaveLane();
            var rowBase = BitwiseAnd(lane, UInt(0xFFFF_FFF0));
            var rowLane = BitwiseAnd(lane, UInt(15));
            var dpp = control.Control;
            inRange = _module.ConstantBool(true);

            if (dpp <= 0xFF)
            {
                var quadLane = BitwiseAnd(lane, UInt(3));
                var selected = UInt(dpp & 3);
                for (var index = 1u; index < 4; index++)
                {
                    selected = _module.AddInstruction(
                        SpirvOp.Select,
                        _uintType,
                        _module.AddInstruction(
                            SpirvOp.IEqual,
                            _boolType,
                            quadLane,
                            UInt(index)),
                        UInt((dpp >> checked((int)(index * 2))) & 3),
                        selected);
                }

                targetLane = IAdd(BitwiseAnd(lane, UInt(0xFFFF_FFFC)), selected);
                return;
            }

            if (dpp is >= 0x101 and <= 0x10F)
            {
                var shift = UInt(dpp & 15);
                var shifted = IAdd(rowLane, shift);
                inRange = _module.AddInstruction(
                    SpirvOp.ULessThan,
                    _boolType,
                    shifted,
                    UInt(16));
                targetLane = IAdd(rowBase, BitwiseAnd(shifted, UInt(15)));
                return;
            }

            if (dpp is >= 0x111 and <= 0x11F)
            {
                var shift = UInt(dpp & 15);
                inRange = _module.AddInstruction(
                    SpirvOp.UGreaterThanEqual,
                    _boolType,
                    rowLane,
                    shift);
                targetLane = IAdd(
                    rowBase,
                    BitwiseAnd(
                        _module.AddInstruction(SpirvOp.ISub, _uintType, rowLane, shift),
                        UInt(15)));
                return;
            }

            if (dpp is >= 0x121 and <= 0x12F)
            {
                targetLane = IAdd(
                    rowBase,
                    BitwiseAnd(
                        _module.AddInstruction(
                            SpirvOp.ISub,
                            _uintType,
                            rowLane,
                            UInt(dpp & 15)),
                        UInt(15)));
                return;
            }

            targetLane = dpp switch
            {
                0x140 => IAdd(rowBase, _module.AddInstruction(
                    SpirvOp.ISub, _uintType, UInt(15), rowLane)),
                0x141 => IAdd(
                    BitwiseAnd(lane, UInt(0xFFFF_FFF8)),
                    _module.AddInstruction(
                        SpirvOp.ISub,
                        _uintType,
                        UInt(7),
                        BitwiseAnd(lane, UInt(7)))),
                >= 0x150 and <= 0x15F => IAdd(rowBase, UInt(dpp & 15)),
                >= 0x160 and <= 0x16F => IAdd(
                    rowBase,
                    BitwiseXor(rowLane, UInt(dpp & 15))),
                _ => lane,
            };
        }

        private uint IsDppWriteEnabled(Gen5DppControl control)
        {
            GetDppSourceLane(control, out _, out var inRange);
            var lane = GuestWaveLane();
            var row = ShiftRightLogical(lane, UInt(4));
            // RDNA2 BANK_MASK partitions each 16-lane row into four contiguous
            // four-lane banks: [0:3], [4:7], [8:11], [12:15].
            var bank = BitwiseAnd(ShiftRightLogical(lane, UInt(2)), UInt(3));
            var rowEnabled = IsNotZero(BitwiseAnd(
                UInt(control.RowMask),
                ShiftLeftLogical(UInt(1), row)));
            var bankEnabled = IsNotZero(BitwiseAnd(
                UInt(control.BankMask),
                ShiftLeftLogical(UInt(1), bank)));
            var sourceAllowsWrite = control.BoundControl
                ? _module.ConstantBool(true)
                : inRange;
            return _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                rowEnabled,
                _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    bankEnabled,
                    sourceAllowsWrite));
        }

        private uint GetFloatSource(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            uint value;
            if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                operand.Value is >= 128 and <= 192)
            {
                value = Float(operand.Value - 128);
            }
            else if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                     operand.Value is >= 193 and <= 208)
            {
                value = Float(-(operand.Value - 192));
            }
            else
            {
                value = Bitcast(
                    _floatType,
                    GetRawSource(
                        instruction,
                        sourceIndex,
                        applySdwaIntegerModifiers: false));
            }

            uint absoluteMask = 0;
            uint negateMask = 0;
            switch (instruction.Control)
            {
                case Gen5Vop3Control control:
                    absoluteMask = control.AbsoluteMask;
                    negateMask = control.NegateMask;
                    break;
                case Gen5SdwaControl control:
                    absoluteMask = control.AbsoluteMask;
                    negateMask = control.NegateMask;
                    break;
                case Gen5DppControl control:
                    absoluteMask = control.AbsoluteMask;
                    negateMask = control.NegateMask;
                    break;
            }

            if ((absoluteMask & (1u << sourceIndex)) != 0)
            {
                value = Ext(4, _floatType, value);
            }

            if ((negateMask & (1u << sourceIndex)) != 0)
            {
                value = _module.AddInstruction(SpirvOp.FNegate, _floatType, value);
            }

            return value;
        }

        private uint GetRawSource64(
            Gen5ShaderInstruction instruction,
            int sourceIndex)
        {
            var operand = instruction.Sources[sourceIndex];
            if (operand.Kind == Gen5OperandKind.ScalarRegister)
            {
                return LoadS64(operand.Value);
            }

            if (operand.Kind == Gen5OperandKind.VectorRegister)
            {
                var vectorLow = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    LoadV(operand.Value));
                var high = _module.AddInstruction(
                    SpirvOp.UConvert,
                    _ulongType,
                    LoadV(operand.Value + 1));
                high = ShiftLeftLogical64(
                    high,
                    _module.Constant64(_ulongType, 32));
                return _module.AddInstruction(
                    SpirvOp.BitwiseOr,
                    _ulongType,
                    vectorLow,
                    high);
            }

            // Scalar inline negative constants are signed immediates. B64
            // consumers sign-extend them, so -1 denotes a full 64-bit mask.
            if (operand.Kind == Gen5OperandKind.EncodedConstant &&
                operand.Value is >= 193 and <= 208)
            {
                var signed = -(long)(operand.Value - 192);
                return _module.Constant64(_ulongType, unchecked((ulong)signed));
            }

            var low = GetRawSource(instruction, sourceIndex);
            return _module.AddInstruction(SpirvOp.UConvert, _ulongType, low);
        }

        private uint LoadS64(uint register)
        {
            var low = _module.AddInstruction(SpirvOp.UConvert, _ulongType, LoadS(register));
            var high = _module.AddInstruction(
                SpirvOp.UConvert,
                _ulongType,
                LoadS(register + 1));
            high = ShiftLeftLogical64(high, _module.Constant64(_ulongType, 32));
            return _module.AddInstruction(SpirvOp.BitwiseOr, _ulongType, low, high);
        }

        private void StoreS64(uint register, uint value)
        {
            StoreS(
                register,
                _module.AddInstruction(SpirvOp.UConvert, _uintType, value));
            var high = ShiftRightLogical64(
                value,
                _module.Constant64(_ulongType, 32));
            StoreS(
                register + 1,
                _module.AddInstruction(SpirvOp.UConvert, _uintType, high));
        }

        private uint EmitFloatBinary(
            Gen5ShaderInstruction instruction,
            SpirvOp operation,
            bool reverse = false)
        {
            var left = GetFloatSource(instruction, reverse ? 1 : 0);
            var right = GetFloatSource(instruction, reverse ? 0 : 1);
            return EmitFloatResult(
                instruction,
                _module.AddInstruction(operation, _floatType, left, right));
        }

        private uint EmitLegacyFloatMultiply(Gen5ShaderInstruction instruction)
        {
            var left = GetFloatSource(instruction, 0);
            var right = GetFloatSource(instruction, 1);
            var product = _module.AddInstruction(SpirvOp.FMul, _floatType, left, right);
            return EmitFloatResult(instruction, ApplyLegacyZeroProduct(left, right, product));
        }

        private uint EmitMullitF32(Gen5ShaderInstruction instruction)
        {
            var left = GetFloatSource(instruction, 0);
            var right = GetFloatSource(instruction, 1);
            var product = _module.AddInstruction(SpirvOp.FMul, _floatType, left, right);
            // The ISA specifies 0.0*x = 0.0; preserve the sign-insensitive
            // zero rule while leaving the documented multiply special values
            // to the target's normal floating-point operation.
            return EmitFloatResult(instruction, ApplyLegacyZeroProduct(left, right, product));
        }

        private uint EmitLegacyFloatMultiplyAccumulate(
            Gen5ShaderInstruction instruction,
            uint destination)
        {
            var left = GetFloatSource(instruction, 0);
            var right = GetFloatSource(instruction, 1);
            var product = _module.AddInstruction(SpirvOp.FMul, _floatType, left, right);
            var addend = Bitcast(_floatType, LoadV(destination));
            return EmitFloatResult(
                instruction,
                _module.AddInstruction(
                    SpirvOp.FAdd,
                    _floatType,
                    ApplyLegacyZeroProduct(left, right, product),
                    addend));
        }

        private uint EmitLegacyFloatMad(Gen5ShaderInstruction instruction)
        {
            var left = GetFloatSource(instruction, 0);
            var right = GetFloatSource(instruction, 1);
            var product = _module.AddInstruction(SpirvOp.FMul, _floatType, left, right);
            return EmitFloatResult(
                instruction,
                _module.AddInstruction(
                    SpirvOp.FAdd,
                    _floatType,
                    ApplyLegacyZeroProduct(left, right, product),
                    GetFloatSource(instruction, 2)));
        }

        private uint ApplyLegacyZeroProduct(uint left, uint right, uint product)
        {
            var leftBits = Bitcast(_uintType, left);
            var rightBits = Bitcast(_uintType, right);
            var leftZero = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                BitwiseAnd(leftBits, UInt(0x7FFF_FFFF)),
                UInt(0));
            var rightZero = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                BitwiseAnd(rightBits, UInt(0x7FFF_FFFF)),
                UInt(0));
            var zeroProduct = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                leftZero,
                rightZero);
            return Bitcast(
                _floatType,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    zeroProduct,
                    UInt(0),
                    Bitcast(_uintType, product)));
        }

        private uint EmitFloatExtBinary(
            Gen5ShaderInstruction instruction,
            uint operation) =>
            EmitFloatResult(
                instruction,
                Ext(
                    operation,
                    _floatType,
                    GetFloatSource(instruction, 0),
                    GetFloatSource(instruction, 1)));

        private uint EmitFloatTernaryExt(
            Gen5ShaderInstruction instruction,
            uint operation)
        {
            var first = Ext(
                operation,
                _floatType,
                GetFloatSource(instruction, 0),
                GetFloatSource(instruction, 1));
            return EmitFloatResult(
                instruction,
                Ext(operation, _floatType, first, GetFloatSource(instruction, 2)));
        }

        private uint EmitIntegerBinary(
            Gen5ShaderInstruction instruction,
            SpirvOp operation,
            bool reverse = false)
        {
            var left = GetRawSource(instruction, reverse ? 1 : 0);
            var right = GetRawSource(instruction, reverse ? 0 : 1);
            if (operation == SpirvOp.ShiftLeftLogical)
            {
                return ShiftLeftLogical(left, right);
            }

            if (operation == SpirvOp.ShiftRightLogical)
            {
                return ShiftRightLogical(left, right);
            }

            if (operation == SpirvOp.ShiftRightArithmetic)
            {
                return ShiftRightArithmetic(left, right);
            }

            return _module.AddInstruction(operation, _uintType, left, right);
        }

        private uint EmitSigned24Product(
            Gen5ShaderInstruction instruction,
            bool high)
        {
            uint SignExtend24(uint value) =>
                ShiftRightArithmetic(ShiftLeftLogical(value, UInt(8)), UInt(8));

            var left = _module.AddInstruction(
                SpirvOp.SConvert,
                _longType,
                Bitcast(_intType, SignExtend24(GetRawSource(instruction, 0))));
            var right = _module.AddInstruction(
                SpirvOp.SConvert,
                _longType,
                Bitcast(_intType, SignExtend24(GetRawSource(instruction, 1))));
            var product = _module.AddInstruction(
                SpirvOp.IMul,
                _longType,
                left,
                right);
            if (high)
            {
                product = _module.AddInstruction(
                    SpirvOp.ShiftRightArithmetic,
                    _longType,
                    product,
                    _module.Constant64(_longType, 32));
            }

            return _module.AddInstruction(SpirvOp.UConvert, _uintType, product);
        }

        private uint EmitLerpU8(Gen5ShaderInstruction instruction)
        {
            var first = GetRawSource(instruction, 0);
            var second = GetRawSource(instruction, 1);
            var rounding = GetRawSource(instruction, 2);
            uint ByteAverage(uint shift)
            {
                var left = BitwiseAnd(ShiftRightLogical(first, UInt(shift)), UInt(0xFF));
                var right = BitwiseAnd(ShiftRightLogical(second, UInt(shift)), UInt(0xFF));
                var roundBit = BitwiseAnd(ShiftRightLogical(rounding, UInt(shift)), UInt(1));
                var sum = IAdd(IAdd(left, right), roundBit);
                return ShiftLeftLogical(
                    BitwiseAnd(ShiftRightLogical(sum, UInt(1)), UInt(0xFF)),
                    UInt(shift));
            }

            return BitwiseOr(
                BitwiseOr(ByteAverage(0), ByteAverage(8)),
                BitwiseOr(ByteAverage(16), ByteAverage(24)));
        }

        private enum CubeCoordinate
        {
            Id,
            Sc,
            Tc,
            Ma,
        }

        private uint EmitCvtOffF32I4(Gen5ShaderInstruction instruction)
        {
            var index = BitwiseAnd(GetRawSource(instruction, 0), UInt(15));
            ReadOnlySpan<float> table =
            [
                0.0f,
                0.0625f,
                0.1250f,
                0.1875f,
                0.2500f,
                0.3125f,
                0.3750f,
                0.4375f,
                -0.5000f,
                -0.4375f,
                -0.3750f,
                -0.3125f,
                -0.2500f,
                -0.1875f,
                -0.1250f,
                -0.0625f,
            ];

            var result = UInt(BitConverter.SingleToUInt32Bits(table[^1]));
            for (var tableIndex = table.Length - 2; tableIndex >= 0; tableIndex--)
            {
                var matches = _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    index,
                    UInt((uint)tableIndex));
                result = _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    matches,
                    UInt(BitConverter.SingleToUInt32Bits(table[tableIndex])),
                    result);
            }

            return result;
        }

        private uint EmitCubeCoordinate(
            Gen5ShaderInstruction instruction,
            CubeCoordinate coordinate)
        {
            var x = GetFloatSource(instruction, 0);
            var y = GetFloatSource(instruction, 1);
            var z = GetFloatSource(instruction, 2);
            var nx = _module.AddInstruction(SpirvOp.FNegate, _floatType, x);
            var ny = _module.AddInstruction(SpirvOp.FNegate, _floatType, y);
            var nz = _module.AddInstruction(SpirvOp.FNegate, _floatType, z);
            var ax = Ext(4, _floatType, x);
            var ay = Ext(4, _floatType, y);
            var az = Ext(4, _floatType, z);
            var amaxXY = Ext(40, _floatType, ax, ay);
            var amax = Ext(40, _floatType, az, amaxXY);
            var ma = _module.AddInstruction(
                SpirvOp.FMul,
                _floatType,
                Float(2),
                amax);
            if (coordinate == CubeCoordinate.Ma)
            {
                return EmitFloatResult(instruction, ma);
            }

            var isZMax = _module.AddInstruction(
                SpirvOp.FOrdGreaterThanEqual,
                _boolType,
                az,
                amaxXY);
            var yGreaterOrEqualX = _module.AddInstruction(
                SpirvOp.FOrdGreaterThanEqual,
                _boolType,
                ay,
                ax);
            var isYMax = _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                _module.AddInstruction(SpirvOp.LogicalNot, _boolType, isZMax),
                yGreaterOrEqualX);
            if (coordinate == CubeCoordinate.Id)
            {
                var isZNeg = _module.AddInstruction(
                    SpirvOp.FOrdLessThan,
                    _boolType,
                    z,
                    Float(0));
                var isYNeg = _module.AddInstruction(
                    SpirvOp.FOrdLessThan,
                    _boolType,
                    y,
                    Float(0));
                var isXNeg = _module.AddInstruction(
                    SpirvOp.FOrdLessThan,
                    _boolType,
                    x,
                    Float(0));
                var zCase = _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    isZNeg,
                    Float(5),
                    Float(4));
                var yCase = _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    isYNeg,
                    Float(3),
                    Float(2));
                var xCase = _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    isXNeg,
                    Float(1),
                    Float(0));
                var xyCase = _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    yGreaterOrEqualX,
                    yCase,
                    xCase);
                return EmitFloatResult(
                    instruction,
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _floatType,
                        isZMax,
                        zCase,
                        xyCase));
            }

            if (coordinate == CubeCoordinate.Sc)
            {
                var isZNeg = _module.AddInstruction(
                    SpirvOp.FOrdLessThan,
                    _boolType,
                    z,
                    Float(0));
                var isXNeg = _module.AddInstruction(
                    SpirvOp.FOrdLessThan,
                    _boolType,
                    x,
                    Float(0));
                var zCase = _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    isZNeg,
                    nx,
                    x);
                var xCase = _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    isXNeg,
                    z,
                    nz);
                var nonZCase = _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    isYMax,
                    x,
                    xCase);
                return EmitFloatResult(
                    instruction,
                    _module.AddInstruction(
                        SpirvOp.Select,
                        _floatType,
                        isZMax,
                        zCase,
                        nonZCase));
            }

            var tcIsYNeg = _module.AddInstruction(
                SpirvOp.FOrdLessThan,
                _boolType,
                y,
                Float(0));
            var tcYCase = _module.AddInstruction(
                SpirvOp.Select,
                _floatType,
                tcIsYNeg,
                nz,
                z);
            return EmitFloatResult(
                instruction,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _floatType,
                    isYMax,
                    tcYCase,
                    ny));
        }

        private uint EmitAddWithCarry(Gen5ShaderInstruction instruction)
        {
            var left = GetRawSource(instruction, 0);
            var right = GetRawSource(instruction, 1);
            var carryMask = instruction.Sources.Count > 2
                ? IsCurrentLaneSet(GetRawSource64(instruction, 2))
                : Load(_boolType, _vcc);
            var carryIn = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                carryMask,
                UInt(1),
                UInt(0));
            var partial = IAdd(left, right);
            var result = IAdd(partial, carryIn);
            var carry = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                _module.AddInstruction(SpirvOp.ULessThan, _boolType, partial, left),
                _module.AddInstruction(SpirvOp.ULessThan, _boolType, result, partial));
            StoreCarryOut(instruction, carry);
            return result;
        }

        private uint EmitSubtractWithBorrow(
            Gen5ShaderInstruction instruction,
            bool reverse)
        {
            var left = GetRawSource(instruction, reverse ? 1 : 0);
            var right = GetRawSource(instruction, reverse ? 0 : 1);
            var borrowMask = instruction.Sources.Count > 2
                ? IsCurrentLaneSet(GetRawSource64(instruction, 2))
                : Load(_boolType, _vcc);
            var borrowIn = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                borrowMask,
                UInt(1),
                UInt(0));
            var partial = _module.AddInstruction(SpirvOp.ISub, _uintType, left, right);
            var result = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                partial,
                borrowIn);
            var borrow = _module.AddInstruction(
                SpirvOp.LogicalOr,
                _boolType,
                _module.AddInstruction(SpirvOp.ULessThan, _boolType, left, right),
                _module.AddInstruction(
                    SpirvOp.ULessThan,
                    _boolType,
                    partial,
                    borrowIn));
            StoreCarryOut(instruction, borrow);
            return result;
        }

        private uint BroadcastFirstWave64Active(uint value)
        {
            var lane = GuestWaveLane();
            EmitConditional(
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    lane,
                    UInt(0)),
                () => Store(WaveBroadcastScratchPointer(), UInt(0)));
            EmitWave64Barrier();

            var activeMask = BooleanToWaveMask(Load(_boolType, _exec));
            var lowMask = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                activeMask);
            var highMask = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(
                    activeMask,
                    _module.Constant64(_ulongType, 32)));
            var hasLow = IsNotZero(lowMask);
            var hasHigh = IsNotZero(highMask);
            var firstLow = Ext(73, _uintType, lowMask);
            var firstHigh = IAdd(UInt(32), Ext(73, _uintType, highMask));
            var firstLane = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                hasLow,
                firstLow,
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    hasHigh,
                    firstHigh,
                    UInt(0)));
            var isFirst = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                lane,
                firstLane);
            EmitConditional(
                _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    isFirst,
                    _module.AddInstruction(
                        SpirvOp.LogicalOr,
                        _boolType,
                        hasLow,
                        hasHigh)),
                () => Store(WaveBroadcastScratchPointer(), value));
            EmitWave64Barrier();
            var result = Load(_uintType, WaveBroadcastScratchPointer());
            EmitWave64Barrier();
            return result;
        }

        private void StoreCarryOut(
            Gen5ShaderInstruction instruction,
            uint carry)
        {
            var activeCarry = _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                Load(_boolType, _exec),
                carry);
            if (instruction.Control is Gen5Vop3Control { ScalarDestination: { } register })
            {
                StoreWaveMask(register, activeCarry);
                return;
            }

            StoreWaveMask(106, activeCarry);
        }

        private bool TryEmitReadlane(
            Gen5ShaderInstruction instruction,
            out string error)
        {
            error = string.Empty;
            if (instruction.Destinations.Count == 0 ||
                instruction.Destinations[0].Kind != Gen5OperandKind.ScalarRegister)
            {
                error = "VReadlaneB32 expects scalar destination";
                return false;
            }

            var destination = instruction.Destinations[0].Value;
            var src0 = GetRawSource(instruction, 0);

            if (_subgroupInvocationIdInput != 0)
            {
                // sdst = vsrc0[lane(src1)] — broadcast from the specified lane.
                var laneSelect = GetRawSource(instruction, 1);
                var broadcast = _module.AddInstruction(
                    SpirvOp.GroupNonUniformBroadcast,
                    _uintType,
                    UInt(3),  // Subgroup scope
                    src0,
                    laneSelect);
                StoreS(destination, broadcast);
            }
            else
            {
                // Fallback: no subgroup ops, read current lane's value.
                StoreS(destination, src0);
            }

            return true;
        }

        private uint EmitPermlane16(
            Gen5ShaderInstruction instruction,
            bool exchangeRows)
        {
            if (instruction.Control is not Gen5Vop3Control control ||
                (control.OperandSelect & ~3u) != 0 ||
                control.AbsoluteMask != 0 ||
                control.NegateMask != 0 ||
                control.OutputModifier != 0 ||
                control.Clamp)
            {
                throw new InvalidOperationException(
                    $"invalid permlane modifiers for {instruction.Opcode}");
            }

            var value = GetRawSource(instruction, 0);
            var selectorLow = GetRawSource(instruction, 1);
            var selectorHigh = GetRawSource(instruction, 2);
            var lane = GuestWaveLane();
            var localLane = BitwiseAnd(lane, UInt(15));
            var lowHalf = _module.AddInstruction(
                SpirvOp.ULessThan,
                _boolType,
                localLane,
                UInt(8));
            var lowShift = ShiftLeftLogical(localLane, UInt(2));
            var highLane = _module.AddInstruction(
                SpirvOp.ISub,
                _uintType,
                localLane,
                UInt(8));
            var highShift = ShiftLeftLogical(highLane, UInt(2));
            var lowSelector = BitwiseAnd(
                ShiftRightLogical(selectorLow, lowShift),
                UInt(15));
            var highSelector = BitwiseAnd(
                ShiftRightLogical(selectorHigh, highShift),
                UInt(15));
            var selector = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                lowHalf,
                lowSelector,
                highSelector);
            var rowBase = BitwiseAnd(lane, UInt(0xFFFF_FFF0));
            if (exchangeRows)
            {
                rowBase = BitwiseXor(rowBase, UInt(16));
            }

            var targetLane = IAdd(rowBase, selector);
            targetLane = BitwiseAnd(targetLane, UInt(31));
            var shuffled = _module.AddInstruction(
                SpirvOp.GroupNonUniformShuffle,
                _uintType,
                UInt(3),
                value,
                targetLane);
            var fetchInactive = (control.OperandSelect & 1) != 0;
            if (fetchInactive)
            {
                return shuffled;
            }

            var activeWord = _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                Load(_boolType, _exec),
                UInt(1),
                UInt(0));
            var sourceActive = IsNotZero(
                _module.AddInstruction(
                    SpirvOp.GroupNonUniformShuffle,
                    _uintType,
                    UInt(3),
                    activeWord,
                    targetLane));
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                sourceActive,
                shuffled,
                UInt(0));
        }

        private uint EmitFloatResult(
            Gen5ShaderInstruction instruction,
            uint value)
        {
            uint outputModifier = 0;
            var clamp = false;
            switch (instruction.Control)
            {
                case Gen5Vop3Control control:
                    outputModifier = control.OutputModifier;
                    clamp = control.Clamp;
                    break;
                case Gen5SdwaControl control:
                    outputModifier = control.OutputModifier;
                    clamp = control.Clamp;
                    break;
            }

            value = outputModifier switch
            {
                1 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(2)),
                2 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(4)),
                3 => _module.AddInstruction(SpirvOp.FMul, _floatType, value, Float(0.5f)),
                _ => value,
            };
            if (clamp)
            {
                value = Ext(43, _floatType, value, Float(0), Float(1));
            }

            return Bitcast(_uintType, value);
        }

        private uint TruncateFloat32ForPack(uint value)
        {
            var raw = BitwiseAnd(
                Bitcast(_uintType, value),
                UInt(0xFFFF_E000));
            return Bitcast(_floatType, raw);
        }

        private uint EmitPermuteByte(uint high, uint low, uint selector)
        {
            uint Select(uint condition, uint whenTrue, uint whenFalse) =>
                _module.AddInstruction(
                    SpirvOp.Select,
                    _uintType,
                    condition,
                    whenTrue,
                    whenFalse);
            uint Equals(uint value) =>
                _module.AddInstruction(
                    SpirvOp.IEqual,
                    _boolType,
                    selector,
                    UInt(value));
            uint SignFill(uint word, uint mask) =>
                Select(
                    _module.AddInstruction(
                        SpirvOp.INotEqual,
                        _boolType,
                        BitwiseAnd(word, UInt(mask)),
                        UInt(0)),
                    UInt(0xFF),
                    UInt(0));

            var source = Select(
                _module.AddInstruction(
                    SpirvOp.ULessThan,
                    _boolType,
                    selector,
                    UInt(4)),
                low,
                high);
            var extracted = BitwiseAnd(
                ShiftRightLogical(
                    source,
                    ShiftLeftLogical(BitwiseAnd(selector, UInt(3)), UInt(3))),
                UInt(0xFF));
            var special = Select(
                Equals(8),
                SignFill(low, 0x00008000),
                Select(
                    Equals(9),
                    SignFill(low, 0x80000000),
                    Select(
                        Equals(10),
                        SignFill(high, 0x00008000),
                        Select(
                            Equals(11),
                            SignFill(high, 0x80000000),
                            Select(Equals(12), UInt(0), UInt(0xFF))))));
            return Select(
                _module.AddInstruction(
                    SpirvOp.ULessThan,
                    _boolType,
                    selector,
                    UInt(8)),
                extracted,
                special);
        }

        private uint Ext(uint operation, uint resultType, params uint[] operands)
        {
            var values = new uint[2 + operands.Length];
            values[0] = _glsl;
            values[1] = operation;
            operands.CopyTo(values, 2);
            return _module.AddInstruction(SpirvOp.ExtInst, resultType, values);
        }

        private uint IsNotZero(uint value) =>
            _module.AddInstruction(SpirvOp.INotEqual, _boolType, value, UInt(0));

        private uint FindMsb64(uint value)
        {
            var low = _module.AddInstruction(SpirvOp.UConvert, _uintType, value);
            var high = _module.AddInstruction(
                SpirvOp.UConvert,
                _uintType,
                ShiftRightLogical64(value, _module.Constant64(_ulongType, 32)));
            var highMsb = Bitcast(_uintType, Ext(75, _intType, Bitcast(_intType, high)));
            var lowMsb = Bitcast(_uintType, Ext(75, _intType, Bitcast(_intType, low)));
            return _module.AddInstruction(
                SpirvOp.Select,
                _uintType,
                IsNotZero(high),
                IAdd(highMsb, UInt(32)),
                lowMsb);
        }

        private uint IsNotZero64(uint value) =>
            _module.AddInstruction(
                SpirvOp.INotEqual,
                _boolType,
                value,
                _module.Constant64(_ulongType, 0));

        private uint SignBit(uint value) =>
            ShiftRightLogical(value, UInt(31));

        private uint SignedAddOverflow(uint left, uint right, uint result)
        {
            var leftSign = SignBit(left);
            var rightSign = SignBit(right);
            var resultSign = SignBit(result);
            var sameSourceSign = _module.AddInstruction(
                SpirvOp.IEqual,
                _boolType,
                leftSign,
                rightSign);
            var resultSignChanged = _module.AddInstruction(
                SpirvOp.INotEqual,
                _boolType,
                leftSign,
                resultSign);
            return _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                sameSourceSign,
                resultSignChanged);
        }

        private uint SignedSubOverflow(uint left, uint right, uint result)
        {
            var leftSign = SignBit(left);
            var rightSign = SignBit(right);
            var resultSign = SignBit(result);
            var differentSourceSign = _module.AddInstruction(
                SpirvOp.INotEqual,
                _boolType,
                leftSign,
                rightSign);
            var resultSignChanged = _module.AddInstruction(
                SpirvOp.INotEqual,
                _boolType,
                leftSign,
                resultSign);
            return _module.AddInstruction(
                SpirvOp.LogicalAnd,
                _boolType,
                differentSourceSign,
                resultSignChanged);
        }

        private static bool TryDecodeInlineConstant(uint encoded, out uint value) =>
            Gen5InlineConstants.TryDecode(encoded, out value);
    }
}
