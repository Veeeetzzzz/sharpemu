// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Ime;

/// <summary>
/// Headless libSceImeDialog. Text is committed into the guest-provided UTF-16
/// buffer; the result structure reports only how the dialog ended.
/// </summary>
public static class ImeDialogExports
{
    private const int StatusNone = 0;
    private const int StatusRunning = 1;
    private const int StatusFinished = 2;

    private const int EndStatusOk = 0;
    private const int EndStatusUserCanceled = 1;
    private const int EndStatusAborted = 2;

    private const int ErrorOk = 0;
    private const int ErrorInvalidAddress = unchecked((int)0x80BC1001);
    private const int ErrorInvalidParam = unchecked((int)0x80BC1002);
    private const int ErrorNotOpened = unchecked((int)0x80BC1003);
    private const int ErrorNotFinished = unchecked((int)0x80BC1004);
    private const int ErrorBusy = unchecked((int)0x80BC1005);

    private const int ParamSize = 0x60;
    private const int ParamMaxTextLengthOffset = 0x24;
    private const int ParamInputTextBufferOffset = 0x28;
    private const int MaxSupportedTextLength = 1024;

    private const string DefaultText = "SharpEmu";

    private static int _status;
    private static int _endStatus;

    [SysAbiExport(
        Nid = "aAx4WY4uwLc",
        ExportName = "sceImeDialogParamInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogParamInit(CpuContext ctx)
    {
        var paramAddress = ctx[CpuRegister.Rdi];
        if (paramAddress == 0)
        {
            return ctx.SetReturn(ErrorInvalidAddress);
        }

        Span<byte> zeroed = stackalloc byte[ParamSize];
        zeroed.Clear();
        if (!ctx.Memory.TryWrite(paramAddress, zeroed))
        {
            Trace($"param_init param=0x{paramAddress:X12} FAILED (unwritable)");
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        Trace($"param_init param=0x{paramAddress:X12}");
        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "NUeBrN7hzf0",
        ExportName = "sceImeDialogInit",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogInit(CpuContext ctx)
    {
        var paramAddress = ctx[CpuRegister.Rdi];
        if (paramAddress == 0)
        {
            Trace("init REJECTED: null param");
            return ctx.SetReturn(ErrorInvalidAddress);
        }

        if (Volatile.Read(ref _status) == StatusRunning)
        {
            Trace($"init REJECTED: busy param=0x{paramAddress:X12}");
            return ctx.SetReturn(ErrorBusy);
        }

        Span<byte> param = stackalloc byte[ParamSize];
        if (!ctx.Memory.TryRead(paramAddress, param))
        {
            Trace($"init REJECTED: param=0x{paramAddress:X12} unreadable");
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var maxTextLength = BinaryPrimitives.ReadUInt32LittleEndian(
            param[ParamMaxTextLengthOffset..]);
        var textBuffer = BinaryPrimitives.ReadUInt64LittleEndian(
            param[ParamInputTextBufferOffset..]);
        if (textBuffer == 0)
        {
            Trace("init REJECTED: inputTextBuffer is null");
            return ctx.SetReturn(ErrorInvalidParam);
        }

        var limit = (int)Math.Min(maxTextLength, MaxSupportedTextLength);
        if (limit == 0)
        {
            Trace("init REJECTED: maxTextLength is zero");
            return ctx.SetReturn(ErrorInvalidParam);
        }

        // Reserve one UTF-16 code unit for the terminator so the guest buffer
        // is never overrun even when maxTextLength is a byte/character bound.
        var text = ResolveText();
        var capacity = limit - 1;
        if (text.Length > capacity)
        {
            text = text[..capacity];
        }

        Span<byte> encoded = stackalloc byte[(text.Length + 1) * sizeof(char)];
        Encoding.Unicode.GetBytes(text, encoded);
        encoded[^2..].Clear();
        if (!ctx.Memory.TryWrite(textBuffer, encoded))
        {
            Trace($"init REJECTED: inputTextBuffer=0x{textBuffer:X12} unwritable");
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        Volatile.Write(ref _endStatus, EndStatusOk);
        Volatile.Write(ref _status, StatusRunning);
        Trace($"init text='{text}' buffer=0x{textBuffer:X16} max={maxTextLength}");
        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "IADmD4tScBY",
        ExportName = "sceImeDialogGetStatus",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogGetStatus(CpuContext ctx)
    {
        // A headless dialog commits immediately once the title polls it.
        var previous = Interlocked.CompareExchange(ref _status, StatusFinished, StatusRunning);
        var current = Volatile.Read(ref _status);
        if (previous != current)
        {
            Trace($"get_status {previous} -> {current}");
        }

        return ctx.SetReturn(current);
    }

    [SysAbiExport(
        Nid = "x01jxu+vxlc",
        ExportName = "sceImeDialogGetResult",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogGetResult(CpuContext ctx)
    {
        var resultAddress = ctx[CpuRegister.Rdi];
        if (resultAddress == 0)
        {
            return ctx.SetReturn(ErrorInvalidAddress);
        }

        if (Volatile.Read(ref _status) != StatusFinished)
        {
            Trace($"get_result REJECTED: status={Volatile.Read(ref _status)}");
            return ctx.SetReturn(ErrorNotFinished);
        }

        // Only endStatus is defined here. Writing an assumed result tail can
        // overwrite a caller's stack-local guard.
        Span<byte> endStatus = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(endStatus, Volatile.Read(ref _endStatus));
        if (!ctx.Memory.TryWrite(resultAddress, endStatus))
        {
            return ctx.SetReturn((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        Trace($"get_result end_status={Volatile.Read(ref _endStatus)}");
        return ctx.SetReturn(ErrorOk);
    }

    [SysAbiExport(
        Nid = "oBmw4xrmfKs",
        ExportName = "sceImeDialogAbort",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogAbort(CpuContext ctx) => Close(ctx, EndStatusAborted);

    [SysAbiExport(
        Nid = "bX4H+sxPI-o",
        ExportName = "sceImeDialogForceClose",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogForceClose(CpuContext ctx) => Close(ctx, EndStatusUserCanceled);

    [SysAbiExport(
        Nid = "gyTyVn+bXMw",
        ExportName = "sceImeDialogTerm",
        Target = Generation.Gen4 | Generation.Gen5,
        LibraryName = "libSceImeDialog")]
    public static int ImeDialogTerm(CpuContext ctx)
    {
        if (Interlocked.Exchange(ref _status, StatusNone) == StatusNone)
        {
            return ctx.SetReturn(ErrorNotOpened);
        }

        Trace("term");
        return ctx.SetReturn(ErrorOk);
    }

    private static int Close(CpuContext ctx, int endStatus)
    {
        if (Interlocked.CompareExchange(ref _status, StatusFinished, StatusRunning) != StatusRunning)
        {
            return ctx.SetReturn(ErrorNotOpened);
        }

        Volatile.Write(ref _endStatus, endStatus);
        Trace($"close end_status={endStatus}");
        return ctx.SetReturn(ErrorOk);
    }

    private static string ResolveText()
    {
        var configured = Environment.GetEnvironmentVariable("SHARPEMU_IME_TEXT");
        return string.IsNullOrEmpty(configured) ? DefaultText : configured;
    }

    internal static void ResetForTests()
    {
        Volatile.Write(ref _status, StatusNone);
        Volatile.Write(ref _endStatus, EndStatusOk);
    }

    private static void Trace(string message)
    {
        if (!string.Equals(
                Environment.GetEnvironmentVariable("SHARPEMU_LOG_IME_DIALOG"),
                "1",
                StringComparison.Ordinal))
        {
            return;
        }

        Console.Error.WriteLine($"[LOADER][TRACE] ime_dialog.{message}");
    }
}
