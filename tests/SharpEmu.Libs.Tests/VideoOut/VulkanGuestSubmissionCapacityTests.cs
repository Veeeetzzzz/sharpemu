// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class VulkanGuestSubmissionCapacityTests
{
    [Theory]
    [InlineData(0, 0, 8, true)]
    [InlineData(7, 0, 8, true)]
    [InlineData(8, 0, 8, false)]
    [InlineData(0, 8, 8, false)]
    [InlineData(4, 4, 8, false)]
    [InlineData(3, 4, 8, true)]
    public void CountsAbandonedSubmissionsAgainstInFlightLimit(
        int pending,
        int abandoned,
        int maximum,
        bool expected)
    {
        Assert.Equal(
            expected,
            VulkanVideoPresenter.HasGuestSubmissionCapacity(
                pending,
                abandoned,
                maximum));
    }

    [Theory]
    [InlineData(-1, 0, 8)]
    [InlineData(0, -1, 8)]
    [InlineData(0, 0, 0)]
    public void RejectsInvalidCapacityInputs(int pending, int abandoned, int maximum)
    {
        Assert.False(
            VulkanVideoPresenter.HasGuestSubmissionCapacity(
                pending,
                abandoned,
                maximum));
    }
}
