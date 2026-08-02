// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class VulkanGuestWorkFollowupTests
{
    [Fact]
    public void FollowupWaitRequiresCompletedWork()
    {
        Assert.False(
            VulkanVideoPresenter.ShouldWaitForGuestWorkFollowup(
                completedWork: 0,
                waitMilliseconds: 2,
                nowTicks: 10,
                followupDeadline: 100,
                renderDeadline: 100));
    }

    [Fact]
    public void FollowupWaitHonoursDisabledProbe()
    {
        Assert.False(
            VulkanVideoPresenter.ShouldWaitForGuestWorkFollowup(
                completedWork: 1,
                waitMilliseconds: 0,
                nowTicks: 10,
                followupDeadline: 100,
                renderDeadline: 100));
    }

    [Theory]
    [InlineData(100, 100, 100)]
    [InlineData(101, 100, 200)]
    [InlineData(101, 200, 100)]
    public void FollowupWaitStopsAtEitherDeadline(
        long nowTicks,
        long followupDeadline,
        long renderDeadline)
    {
        Assert.False(
            VulkanVideoPresenter.ShouldWaitForGuestWorkFollowup(
                completedWork: 1,
                waitMilliseconds: 2,
                nowTicks: nowTicks,
                followupDeadline: followupDeadline,
                renderDeadline: renderDeadline));
    }

    [Fact]
    public void FollowupWaitIsAllowedInsideBothBudgets()
    {
        Assert.True(
            VulkanVideoPresenter.ShouldWaitForGuestWorkFollowup(
                completedWork: 1,
                waitMilliseconds: 2,
                nowTicks: 10,
                followupDeadline: 100,
                renderDeadline: 200));
    }
}
