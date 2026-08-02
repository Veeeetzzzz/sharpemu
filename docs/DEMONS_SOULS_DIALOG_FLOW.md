# Demon's Souls dialog flow

The save-data dialog is headless in this build, but its state transitions and
result layout still have to match what the game expects. The implementation
now preserves the observable sequence

```text
initialize -> open -> RUNNING -> FINISHED -> get result
```

Two details are load-bearing for the Demon's Souls path:

- repeated initialization is accepted without resetting a live dialog;
- `OrbisSaveDataDialogParam` stores the dialog mode at `+0x34`, after the
  common header size at `+0x00`, and reads optional user data only when the
  declared structure covers that field.

The first status poll remains `RUNNING` and the second finishes the headless
dialog. This avoids the title rejecting a result because it never observed a
live dialog, or reopening a dialog forever after echoing the header size as the
mode.

The behavior is based on the Demon's Souls-specific evidence and regression
tests in upstream [#730](https://github.com/sharpemu/sharpemu/pull/730), while
leaving its unrelated GUI and stale-base renderer changes out of this
checkpoint.
