# Oversized detile work ownership

SharpEmu can bind one guest image through several shader slots in the same
translated draw or compute dispatch. The bindings must remain distinct because
their sampler, view, storage, and mip state can differ, but the immutable host
source bytes do not need to be copied once per binding.

The source-sharing policy follows the investigation in upstream
[#731](https://github.com/sharpemu/sharpemu/pull/731) and the allocation/FPS
regression tracked in [#618](https://github.com/sharpemu/sharpemu/issues/618):

- the reuse table is scoped to one translated draw/dispatch;
- a source is shared only when the complete guest descriptor, selected mip,
  array view, storage mode, output layout, representation (linear or tiled),
  byte length, and guest write generation match;
- every binding still receives its own `GuestDrawTexture` record;
- empty/fallback sources are never entered in the table; and
- the source arrays are treated as immutable after AGC translation.

This reduces duplicate managed-array ownership for repeated oversized detile
bindings without changing Vulkan/Metal view creation or queue ordering. It does
not attempt to share backend images or descriptors; those remain the
responsibility of the presenter caches.
