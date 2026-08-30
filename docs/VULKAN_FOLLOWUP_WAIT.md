# Vulkan guest-work follow-up wait

Guest GPU work is produced asynchronously from the presenter thread. A render
tick can drain the queue immediately before the producer publishes the next
draw or ordered flip. If the presenter exits the drain at that instant, it
may sample the previous image and leave the window on a splash or black frame
until the next tick.

The presenter now performs a condition-variable probe after it has completed at
least one item but finds no ready item. The default probe is 2 ms, with a 24 ms
budget per render tick. The producer pulses the same queue condition when it
enqueues work, so a dependent draw/flip can be consumed in the current tick
without polling or an unbounded sleep.

`SHARPEMU_RENDER_FOLLOWUP_WAIT_MS` changes the individual probe duration and
`SHARPEMU_RENDER_FOLLOWUP_BUDGET_MS` changes the per-tick budget. Set either to
`0` to disable that part of the policy for diagnostics. Coverage for the
deadline policy is in `VulkanGuestWorkFollowupTests`.
