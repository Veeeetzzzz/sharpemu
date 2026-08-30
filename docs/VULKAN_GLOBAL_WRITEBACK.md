# Vulkan guest-buffer writeback

Writable guest buffers are synchronized back to emulated memory after the
corresponding GPU timeline completes. The mapped host page is compared with a
shadow copy, then only changed bytes are overlaid onto a freshly read guest
page. This protects CPU writes in bytes that the shader did not touch.

The comparison is deliberately two-tiered:

- a 128-byte coarse scan classifies a dense page and publishes it as one bounded
  page run;
- sparse pages skip equal spans with `Vector<byte>` and scan only the mismatch
  runs.

When a sparse page has more than 64 runs, the writeback overlays the span from
the first to last run as one GPU-owned region before a single guest-page write.
Pages below that threshold overlay only the exact changed runs onto the live
guest page, while the threshold bounds bookkeeping for heavily fragmented GPU
output.
Unreadable pages retain the existing small-gap coalescing and exact-run
fallback, because no full-page merge is safe without the live guest bytes.

This keeps writeback correctness unchanged while avoiding a byte-at-a-time scan
and a large number of tiny writes on fragmented GPU output. Regression coverage
is in `VulkanWritebackScanTests`.
