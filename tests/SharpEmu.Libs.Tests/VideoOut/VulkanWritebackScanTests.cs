// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class VulkanWritebackScanTests
{
    [Fact]
    public void SparsePageReturnsOnlyChangedRunsAndPreservesAbsoluteOffset()
    {
        var mapped = new byte[2048];
        var shadow = new byte[2048];
        mapped[3] = 1;
        mapped[4] = 2;
        mapped[300] = 3;
        mapped[301] = 4;
        var runs = new List<(int Start, int Length)>();

        var changedBytes = VulkanVideoPresenter.ScanChangedWritebackRuns(
            mapped,
            shadow,
            absoluteOffset: 0x2000,
            runs);

        Assert.Equal(4, changedBytes);
        Assert.Equal(
            [
                (0x2003, 2),
                (0x212C, 2),
            ],
            runs);
    }

    [Fact]
    public void DensePageCollapsesToOneFullPageRun()
    {
        var mapped = new byte[1024];
        var shadow = new byte[1024];
        for (var block = 0; block < 4; block++)
        {
            mapped[block * 128] = 1;
        }

        var runs = new List<(int Start, int Length)>();
        var changedBytes = VulkanVideoPresenter.ScanChangedWritebackRuns(
            mapped,
            shadow,
            absoluteOffset: 0x4000,
            runs);

        Assert.Equal(mapped.Length, changedBytes);
        Assert.Equal([(0x4000, mapped.Length)], runs);
    }

    [Fact]
    public void EqualPageProducesNoRuns()
    {
        var mapped = Enumerable.Repeat((byte)0xA5, 257).ToArray();
        var shadow = mapped.ToArray();
        var runs = new List<(int Start, int Length)> { (1, 2) };

        var changedBytes = VulkanVideoPresenter.ScanChangedWritebackRuns(
            mapped,
            shadow,
            absoluteOffset: 17,
            runs);

        Assert.Equal(0, changedBytes);
        Assert.Empty(runs);
    }

    [Fact]
    public void MismatchedSpansAreRejected()
    {
        Assert.Throws<ArgumentException>(() =>
            VulkanVideoPresenter.ScanChangedWritebackRuns(
                new byte[4],
                new byte[3],
                absoluteOffset: 0,
                new List<(int Start, int Length)>()));
    }
}
