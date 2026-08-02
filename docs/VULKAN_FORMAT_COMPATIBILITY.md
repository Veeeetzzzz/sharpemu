# Vulkan guest-image format compatibility

Guest images are created with `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` so a
render target can be viewed through another Vulkan format when the two formats
belong to the same format-compatibility class. The presenter keeps a stricter
identity-alias policy for numeric reinterpretations, but the underlying view
class still needs to include every same-layout format that the renderer may
request.

The compatibility table now covers the missing one- and two-channel sRGB
formats, signed-normalized 16-bit pairs, packed B8G8R8A8/A8B8G8R8 variants,
E5B9G9R9, and the signed-normalized 16-bit four-channel format. This prevents
an otherwise compatible format transition from destroying the existing guest
image and recreating an empty allocation.

Compressed formats remain excluded from this table. Numeric reinterpretations
that share a texel width are still intentionally recreated rather than aliased,
because the translated pipeline's numeric interpretation must match the
attachment format.

The table follows the format-compatibility-class definitions in the
[Vulkan specification](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#formats-compatibility-classes)
and the focused behavior in upstream
[#747](https://github.com/sharpemu/sharpemu/pull/747).
