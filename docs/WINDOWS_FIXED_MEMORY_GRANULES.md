# Windows fixed guest-memory granules

Windows reserves virtual address space at 64 KiB allocation-granule boundaries,
while the guest mapping interface exposes 4 KiB pages. A sequence of adjacent
fixed guest mappings can therefore fail if each call asks the host for a new
reservation: the first call owns the 64 KiB granule and the second call is
rejected even though its guest pages are still free.

`PhysicalVirtualMemory` reserves each required host-granule range once and
tracks its allocation base in `_fixedGranuleReservationBases`. Each mapping
commits only the requested page range, so adjacent `AllocateAt` and
`TryBackFixedRange` calls can share a reservation without exposing uncommitted
pages to the guest. Existing tracked regions are also accepted as trusted
allocation owners.

The fixed-allocation gate serializes reservation ownership and rollback.
Reservations created by a failed transaction are released, while reservations
that predate it remain available to adjacent mappings. `Clear` releases both
ordinary regions and shared granule reservations exactly once.

Regression coverage is in
`tests/SharpEmu.Libs.Tests/Memory/GuestMemoryAllocatorTests.cs`; the two granule
tests run on Windows and the existing fake-host rollback tests cover failed and
partially occupied ranges.
