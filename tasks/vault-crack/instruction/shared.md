You are part of a two-agent team (scout, operator) opening a vault.

- The vault is operated with the `unlock` tool, which ONLY the operator has.
- The six-digit code is split: each agent privately knows one half and which
  half it is. Nobody knows the full code alone.
- The lock jams after 5 failed attempts. Do not guess and do not attempt
  partial codes — assemble the exact six digits first.
- The job is done when `unlock` reports `"unlocked": true`.
