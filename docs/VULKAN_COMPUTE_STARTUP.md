# Vulkan compute startup

`SubmitComputeDispatch` can be the first GPU submission made by a title. The
presenter worker must therefore start at the same lifecycle boundary as
draw/presentation submissions; otherwise the dispatch is queued with no
consumer and later waits can stall indefinitely.

The startup guard is deliberately small and idempotent:

- closed presenters never start;
- an existing presenter thread is left alone; and
- an open presenter with no consumer starts exactly once.

The dispatch is validated before this lifecycle check, so empty or no-op
compute work does not create a window. The presenter is started while holding
the shared gate after valid work and its storage-image dependencies have been
published.

This behavior originated in upstream PR #747. Vulkan queue ordering and
storage-image bookkeeping are otherwise independent of the startup policy.
