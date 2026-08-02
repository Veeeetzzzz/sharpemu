// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Agc;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

public sealed class GuestTextureSourceReuseTests
{
    [Fact]
    public void Share_ReusesExactSourceButKeepsIncompatibleViewsSeparate()
    {
        var cache = new GuestTextureSourceCache();
        var key = CreateKey();
        var first = new byte[32];
        var second = new byte[32];

        Assert.Same(first, cache.Share(key, first));
        Assert.Same(first, cache.Share(key, second));

        var differentMip = key with { MipLevel = key.MipLevel + 1 };
        Assert.Same(second, cache.Share(differentMip, second));
        Assert.Equal(2, cache.Count);
    }

    [Fact]
    public void Share_DoesNotAliasTiledAndLinearRepresentations()
    {
        var cache = new GuestTextureSourceCache();
        var linearKey = CreateKey(GuestTextureSourceKind.LinearPixels);
        var tiledKey = CreateKey(GuestTextureSourceKind.TiledBytes);
        var linear = new byte[64];
        var tiled = new byte[64];

        Assert.Same(linear, cache.Share(linearKey, linear));
        Assert.Same(tiled, cache.Share(tiledKey, tiled));
        Assert.Equal(2, cache.Count);
    }

    [Fact]
    public void Share_LeavesEmptySourcesUntouched()
    {
        var cache = new GuestTextureSourceCache();
        var empty = Array.Empty<byte>();

        Assert.Same(empty, cache.Share(CreateKey(), empty));
        Assert.Equal(0, cache.Count);
    }

    private static GuestTextureSourceReuseKey CreateKey(
        GuestTextureSourceKind kind = GuestTextureSourceKind.LinearPixels) =>
        new(
            Kind: kind,
            Descriptor: new GuestTextureSourceDescriptorIdentity(
                Address: 0x102A400000,
                Width: 1024,
                Height: 1024,
                Format: 4,
                NumberType: 7,
                TileMode: 24,
                Type: 13,
                BaseLevel: 0,
                LastLevel: 0,
                Pitch: 1024,
                DstSelect: 0xFAC,
                Depth: 80,
                BaseArray: 0,
                ArrayPitch: 0,
                MaxMip: 0,
                MinLod: 0,
                MinLodWarn: 0,
                BcSwizzle: 0,
                MetadataAddress: 0,
                DescriptorFlags: 0,
                HasExtendedDescriptor: false),
            IsStorage: false,
            MipLevel: 0,
            IsArrayed: true,
            ArrayLayers: 80,
            OutputWidth: 1024,
            OutputHeight: 1024,
            OutputPitch: 1024,
            OutputMipLevels: 1,
            OutputBaseMipLevel: 0,
            OutputResourceMipLevels: 1,
            OutputDepth: 80,
            WriteGeneration: 12,
            SourceLength: 32);
}
