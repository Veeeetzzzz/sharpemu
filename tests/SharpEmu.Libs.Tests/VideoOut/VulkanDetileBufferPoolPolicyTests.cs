// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class VulkanDetileBufferPoolPolicyTests
{
    [Theory]
    [InlineData(0, 64, 128, true)]
    [InlineData(64, 64, 128, true)]
    [InlineData(65, 64, 128, false)]
    [InlineData(0, 129, 128, false)]
    [InlineData(18446744073709551614UL, 1, ulong.MaxValue, true)]
    [InlineData(ulong.MaxValue, 1, ulong.MaxValue, false)]
    [InlineData(ulong.MaxValue, 2, ulong.MaxValue, false)]
    public void AppliesByteBudgetWithoutUnsignedOverflow(
        ulong cached,
        ulong allocation,
        ulong maximum,
        bool expected)
    {
        Assert.Equal(
            expected,
            VulkanDetilePass.CanCacheBuffer(cached, allocation, maximum));
    }
}
