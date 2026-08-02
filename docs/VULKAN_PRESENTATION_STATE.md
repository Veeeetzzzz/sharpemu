# Vulkan presentation state

Taking a queued presentation is a consumption event even when the frame is
not submitted to the swapchain. A malformed/empty presentation, an oversized
staging upload, or an uninitialized guest image can be dropped after
`TryTakePresentation` succeeds. Those paths must still advance the presented
sequence; otherwise the render loop repeatedly revisits the same dropped flip
and can appear stuck on a black screen.

The sequence is intentionally *not* advanced for an acquire timeout or a
surface-out-of-date result: those paths release the frame resources and retry
the same presentation after swapchain recovery. This keeps retryable Vulkan
conditions distinct from terminal presentation drops.

This follows the state-consistency portion of upstream
[#747](https://github.com/sharpemu/sharpemu/pull/747) without importing its
unrelated renderer changes.
