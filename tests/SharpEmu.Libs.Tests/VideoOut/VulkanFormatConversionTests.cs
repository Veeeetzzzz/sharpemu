// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.VideoOut;
using Silk.NET.Vulkan;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class VulkanFormatConversionTests
{
    [Theory]
    [InlineData(Format.R8G8B8A8Unorm, Format.A2R10G10B10UnormPack32)]
    [InlineData(Format.R8G8B8A8Unorm, Format.A2B10G10R10UnormPack32)]
    [InlineData(Format.A2R10G10B10UnormPack32, Format.R8G8B8A8Unorm)]
    [InlineData(Format.A2B10G10R10UnormPack32, Format.R8G8B8A8Unorm)]
    public void PackedTenBitSwapRequiresRealConversion(Format from, Format to)
    {
        Assert.True(
            VulkanVideoPresenter.IsCompatibleGuestImageViewFormat(from, to));
        Assert.True(VulkanVideoPresenter.RequiresRealFormatConversion(from, to));
    }

    [Theory]
    [InlineData(Format.R8G8B8A8Unorm, Format.B8G8R8A8Unorm)]
    [InlineData(Format.R8G8B8A8Unorm, Format.R8G8B8A8Srgb)]
    [InlineData(Format.A2R10G10B10UnormPack32, Format.A2B10G10R10UnormPack32)]
    [InlineData(Format.R16G16B16A16Sfloat, Format.R32G32Sfloat)]
    public void OtherCompatibleOrNumericPairsRemainRawViewOnly(Format from, Format to)
    {
        Assert.False(VulkanVideoPresenter.RequiresRealFormatConversion(from, to));
    }

    [Fact]
    public void RawRgba8BitsExplainTheObservedTenBitRedCast()
    {
        const uint opaqueBlackRgba8 = 0xFF000000u;

        var alpha2Bit = (opaqueBlackRgba8 >> 30) & 0x3u;
        var red10Bit = (opaqueBlackRgba8 >> 20) & 0x3FFu;
        var green10Bit = (opaqueBlackRgba8 >> 10) & 0x3FFu;
        var blue10Bit = opaqueBlackRgba8 & 0x3FFu;

        Assert.Equal(3u, alpha2Bit);
        Assert.Equal(1008u, red10Bit);
        Assert.Equal(0u, green10Bit);
        Assert.Equal(0u, blue10Bit);

        var redAsFloat = red10Bit / 1023.0;
        Assert.InRange(redAsFloat, 0.9852, 0.9855);
    }
}
