# Windows fixed guest-memory granules

Windows reserves virtual address space at 64 KiB allocation-granule boundaries,
while the guest mapping interface exposes 4 KiB pages. A sequence of adjacent
fixed guest mappings can therefore fail if each call asks the host for a new
reservation: the first call owns the 64 KiB granule and the second call is
rejected even though its guest pages are still free.

`PhysicalVirtualMemory` now reserves each covering host granule once and tracks
its base in `_fixedGranuleReservationBases`. Each mapping commits only the
requested page range, so adjacent `AllocateAt` and `TryBackFixedRange` calls can
share the reservation without exposing uncommitted pages to the guest. The
fixed-allocation gate is acquired before the region lock, and rollback releases
only reservations created by the failed transaction. `Clear` releases both
ordinary regions and shared granule reservations exactly once.

This is the Windows-specific part of upstream PR #748. Its NORMAL-mutex change
is intentionally not included here: the current branch retains the documented
Gen5 wrapper compatibility behavior in `KernelPthreadCompatExports`.

Regression coverage is in
`tests/SharpEmu.Libs.Tests/Memory/GuestMemoryAllocatorTests.cs`; the two granule
tests run on Windows and the existing fake-host rollback tests remain scoped to
the non-Windows allocation model.
