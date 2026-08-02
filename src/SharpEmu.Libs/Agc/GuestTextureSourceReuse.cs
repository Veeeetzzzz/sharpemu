// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Agc;

/// <summary>
/// Distinguishes the two immutable host-side source representations that can
/// be attached to a <see cref="SharpEmu.Libs.Gpu.GuestDrawTexture"/>.
/// </summary>
internal enum GuestTextureSourceKind : byte
{
    LinearPixels,
    TiledBytes,
}

/// <summary>
/// Raw guest descriptor identity used when sharing a source array within one
/// translated draw/dispatch. Keeping the complete descriptor here prevents a
/// view with a different layout or mip allocation from aliasing another view.
/// </summary>
internal readonly record struct GuestTextureSourceDescriptorIdentity(
    ulong Address,
    uint Width,
    uint Height,
    uint Format,
    uint NumberType,
    uint TileMode,
    uint Type,
    uint BaseLevel,
    uint LastLevel,
    uint Pitch,
    uint DstSelect,
    uint Depth,
    uint BaseArray,
    uint ArrayPitch,
    uint MaxMip,
    uint MinLod,
    uint MinLodWarn,
    uint BcSwizzle,
    ulong MetadataAddress,
    uint DescriptorFlags,
    bool HasExtendedDescriptor);

/// <summary>
/// Exact source ownership identity. The cache is intentionally local to one
/// translated draw/dispatch, so an unknown write generation (-1) is still
/// safe: all bindings are observed while the guest operation is being
/// translated, and no array escapes as a mutable AGC-side buffer.
/// </summary>
internal readonly record struct GuestTextureSourceReuseKey(
    GuestTextureSourceKind Kind,
    GuestTextureSourceDescriptorIdentity Descriptor,
    bool IsStorage,
    uint MipLevel,
    bool IsArrayed,
    uint ArrayLayers,
    uint OutputWidth,
    uint OutputHeight,
    uint OutputPitch,
    uint OutputMipLevels,
    uint OutputBaseMipLevel,
    uint OutputResourceMipLevels,
    uint OutputDepth,
    long WriteGeneration,
    int SourceLength);

/// <summary>
/// Shares immutable source arrays while preserving a distinct
/// <c>GuestDrawTexture</c> record for every binding. Backends only read these
/// arrays; per-binding sampler/view/storage state remains in the record.
/// </summary>
internal sealed class GuestTextureSourceCache
{
    private readonly Dictionary<GuestTextureSourceReuseKey, byte[]> _sources = new();

    public byte[] Share(GuestTextureSourceReuseKey key, byte[] candidate)
    {
        if (candidate.Length == 0)
        {
            return candidate;
        }

        if (_sources.TryGetValue(key, out var existing))
        {
            return existing;
        }

        _sources.Add(key, candidate);
        return candidate;
    }

    internal int Count => _sources.Count;
}
