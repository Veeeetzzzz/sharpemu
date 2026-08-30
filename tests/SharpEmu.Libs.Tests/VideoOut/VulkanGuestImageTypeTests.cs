// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Silk.NET.Vulkan;
using SharpEmu.Libs.VideoOut;
using Xunit;

namespace SharpEmu.Libs.Tests.VideoOut;

public sealed class VulkanGuestImageTypeTests
{
    [Fact]
    public void ThreeDimensionalDescriptorsMapToVolumeImageAndViewTypes()
    {
        Assert.Equal(
            ImageType.Type3D,
            VulkanVideoPresenter.GetGuestTextureImageType(
                VulkanVideoPresenter.Gen5TextureType3D));
        Assert.Equal(
            ImageViewType.Type3D,
            VulkanVideoPresenter.GetGuestTextureViewType(
                VulkanVideoPresenter.Gen5TextureType3D,
                arrayedView: true));
        Assert.Equal(
            7u,
            VulkanVideoPresenter.GetGuestTextureDepth(
                VulkanVideoPresenter.Gen5TextureType3D,
                7));
    }

    [Fact]
    public void TwoDimensionalArraysKeepLayersSeparateFromImageDepth()
    {
        Assert.Equal(
            ImageType.Type2D,
            VulkanVideoPresenter.GetGuestTextureImageType(
                VulkanVideoPresenter.Gen5TextureType2D));
        Assert.Equal(
            ImageViewType.Type2DArray,
            VulkanVideoPresenter.GetGuestTextureViewType(
                VulkanVideoPresenter.Gen5TextureType2D,
                arrayedView: true));
        Assert.Equal(
            1u,
            VulkanVideoPresenter.GetGuestTextureDepth(
                VulkanVideoPresenter.Gen5TextureType2D,
                7));
    }

    [Fact]
    public void TwoDimensionalMipExtentsKeepDepthAtOne()
    {
        var extent = VulkanVideoPresenter.GetGuestImageMipExtent(
            width: 128,
            height: 64,
            depth: 7,
            type: VulkanVideoPresenter.Gen5TextureType2D,
            mipLevel: 1);

        Assert.Equal(64u, extent.Width);
        Assert.Equal(32u, extent.Height);
        Assert.Equal(1u, extent.Depth);
    }

    [Fact]
    public void ThreeDimensionalMipExtentsShrinkDepthWithTheMip()
    {
        var extent = VulkanVideoPresenter.GetGuestImageMipExtent(
            width: 128,
            height: 64,
            depth: 4,
            type: VulkanVideoPresenter.Gen5TextureType3D,
            mipLevel: 1);

        Assert.Equal(64u, extent.Width);
        Assert.Equal(32u, extent.Height);
        Assert.Equal(2u, extent.Depth);
    }
}
