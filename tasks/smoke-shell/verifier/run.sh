#!/bin/sh
# Verifier contract: write reward.json next to this script.
dir="$(dirname "$0")"
content="$(cat /work/hello.txt 2>/dev/null)"
if [ "$content" = "hello sentinel" ]; then
    printf '{"reward": 1.0, "passed": true}\n' > "$dir/reward.json"
else
    printf '{"reward": 0.0, "passed": false, "found": "%s"}\n' "$content" > "$dir/reward.json"
fi
