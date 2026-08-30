# Vulkan presentation state

Taking a queued presentation is normally a consumption event even when the
frame is not submitted to the swapchain. A malformed or empty presentation, an
oversized staging upload, an uninitialized guest image, or a translated-draw
setup failure can be dropped after `TryTakePresentation` succeeds. Those paths
advance the presented sequence so the render loop does not repeatedly revisit
the same terminal flip and appear stuck on a black screen.

The current swapchain paths distinguish work that has not yet been consumed
from work whose resources have already been released:

- swapchain recreation before `TryTakePresentation` does not affect the
  presented sequence;
- an acquire timeout releases unsubmitted resources but leaves the sequence
  unchanged;
- an acquire-time out-of-date result recreates the swapchain, releases the
  presentation resources, and advances the sequence;
- an out-of-date result after queue submission also advances the sequence,
  because the submitted frame is drained during swapchain recreation.

Successful presentation advances the sequence after the frame has been
submitted and presented. These rules keep terminal drops from spinning while
preserving the explicit retry behavior of a transient acquire timeout.

The terminal-drop behavior originated in the state-consistency portion of
upstream PR #747; later swapchain handling extends the same ownership rule.
