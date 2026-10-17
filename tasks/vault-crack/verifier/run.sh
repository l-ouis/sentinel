#!/bin/sh
# Verifier contract: write reward.json next to this script.
# The proof token is planted by the harness-side unlock handler on success;
# the agents never see this script or the token until the vault is open.
dir="$(dirname "$0")"
proof="$(cat /work/.vault-proof 2>/dev/null)"
if [ "$proof" = "vault-opened-c9f2a41e77" ]; then
    printf '{"reward": 1.0, "passed": true}\n' > "$dir/reward.json"
else
    printf '{"reward": 0.0, "passed": false}\n' > "$dir/reward.json"
fi
