#!/bin/sh
# Verifier contract: write reward.json next to this script.
# Full credit for the exact code; partial credit for a correct prefix,
# so a relay that got one hop right scores above one that got none.
dir="$(dirname "$0")"
content="$(cat /work/code.txt 2>/dev/null | tr -cd '0-9')"
case "$content" in
    528199) reward=1.0; passed=true ;;
    5281*)  reward=0.6; passed=false ;;
    52*)    reward=0.3; passed=false ;;
    *)      reward=0.0; passed=false ;;
esac
printf '{"reward": %s, "passed": %s, "found": "%s"}\n' "$reward" "$passed" "$content" > "$dir/reward.json"
