# Vulkan guest-image format conversion

Guest render targets are backed by mutable Vulkan images so a compatible view
can be reused when a title alternates between numeric formats. A compatible
view is not always a compatible value representation: `R8G8B8A8_UNORM` and the
packed `A2R10G10B10_UNORM_PACK32`/`A2B10G10R10_UNORM_PACK32` formats occupy the
same 32 bits but assign different widths and channel positions.

Treating that swap as a raw image-view reinterpretation turns opaque RGBA8 data
into a large ten-bit red value (for example, `0xFF000000` becomes red `1008 /
1023`). That is the colour corruption seen in the noisy-bar frames.

The presenter now keeps the existing image, converts its level-zero contents
with a Vulkan `CmdBlitImage` through typed transfer scratch images, and then
creates the requested view and render-pass attachments. The conversion is
ordered after any open batch and its fence is waited before scratch resources
are destroyed. Non-bit-incompatible view pairs retain the cheaper alias path.

Conversion is enabled by default for these known pairs. Set
`SHARPEMU_DISABLE_REAL_FORMAT_CONVERSION=1` only when diagnosing a device or
driver-specific failure; this restores the previous raw-view behavior.

Policy coverage is in `VulkanFormatConversionTests`.
