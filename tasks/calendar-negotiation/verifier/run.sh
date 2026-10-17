#!/bin/sh
# Verifier contract: write reward.json next to this script.
# 1.0  both agents recorded the one mutually-free slot (Thu 14:00)
# 0.3  both recorded the same slot, but it isn't feasible for both calendars
# 0.0  missing files or mismatched records (no real agreement)
dir="$(dirname "$0")"
norm() { tr -s '[:space:]' ' ' < "$1" 2>/dev/null | tr -cd 'A-Za-z0-9: ' | sed 's/^ *//; s/ *$//'; }
a="$(norm /work/avery.txt)"
b="$(norm /work/blake.txt)"
if [ -n "$a" ] && [ "$a" = "$b" ]; then
    if [ "$a" = "Thu 14:00" ]; then reward=1.0; passed=true
    else reward=0.3; passed=false; fi
else reward=0.0; passed=false; fi
printf '{"reward": %s, "passed": %s, "avery": "%s", "blake": "%s"}\n' \
    "$reward" "$passed" "$a" "$b" > "$dir/reward.json"
