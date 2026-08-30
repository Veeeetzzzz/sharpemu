# Demon's Souls IME flow

The game’s name-entry path supplies an `OrbisImeDialogParam` block and reads
the committed UTF-16 text from its `inputTextBuffer`. `sceImeDialogGetResult`
only reports the end status; it is not the place to return the name.

The headless implementation therefore follows this contract:

```text
param-init -> init (write UTF-16 text) -> RUNNING -> FINISHED -> get-result
```

It bounds the write to `maxTextLength - 1` characters, reserves the terminator,
rejects null/invalid buffers, preserves the result tail supplied by the caller,
and exposes abort/force-close status codes. `SHARPEMU_IME_TEXT` can provide a
deterministic test name; otherwise the fallback is `SharpEmu`.

This maps the Demon's Souls-specific IME evidence and regression coverage from
upstream [#730](https://github.com/sharpemu/sharpemu/pull/730) without importing
its unrelated stale-base changes.
