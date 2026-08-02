# Upstream export compatibility

Upstream PR #749 identified three imports that recent Gen5 titles can resolve
before they reach their normal runtime path:

- `sceKernelIsTrinityMode` (`tU5e3f9gSiU`) returns a deterministic base-model
  result (`0`) because SharpEmu does not emulate the PS5 Pro/Trinity profile.
- `sceKernelGetOpenPsId` (`DLORcroUqbc`) validates and writes the 16-byte
  Open-PSID output buffer. The current compatibility identity is zeroed, but
  invalid or unreadable guest pointers still return the correct error.
- `sceNpTrophy2GetTrophyInfoArray` (`y3zHpdZO6ME`) is registered and returns
  `ORBIS_GEN2_ERROR_NOT_FOUND`, matching the existing single-trophy query until
  trophy storage is implemented.

Registering these stubs is important even when their backing services are not
implemented: an unresolved NID can leave a title in an import retry loop before
the graphics path is reached. Contract tests cover the registry-facing return
values and the guest-memory error behavior.
