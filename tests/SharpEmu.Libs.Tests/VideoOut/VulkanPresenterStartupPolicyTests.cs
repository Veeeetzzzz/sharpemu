// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class VulkanPresenterStartupPolicyTests
{
    [Theory]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    public void StartsOnlyWhenOpenAndNoConsumerExists(
        bool closed,
        bool threadActive,
        bool expected)
    {
        Assert.Equal(
            expected,
            VulkanVideoPresenter.ShouldStartPresenter(closed, threadActive));
    }
}
