// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.Libs.Tests.Ampr;

/// <summary>
/// AMPR tests share the process-global file-id registry and guest mount table.
/// Keep the classes in one non-parallel collection so a test cleanup cannot
/// erase another test's file mapping midway through a command-buffer call.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class AmprStateCollection
{
    public const string Name = "AmprState";
}
