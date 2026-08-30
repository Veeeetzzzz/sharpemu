# Vulkan compute startup

`SubmitComputeDispatch` can be the first GPU submission made by a title. The
presenter worker must therefore be started at the same lifecycle boundary as
draw/presentation submissions; otherwise the dispatch is queued with no
consumer and later waits can stall indefinitely.

The startup guard is deliberately small and idempotent:

- closed presenters never start;
- an existing presenter thread is left alone; and
- an open presenter with no consumer starts exactly once.

This follows the behavior documented in upstream
[#747](https://github.com/sharpemu/sharpemu/pull/747), while keeping the
checkpoint limited to the compute-submission lifecycle fix. The presenter
implementation, Vulkan queue ordering, and storage-image bookkeeping are
otherwise unchanged.
