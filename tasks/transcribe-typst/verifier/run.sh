#!/bin/sh
dir="$(dirname "$0")"
python3 "$dir/grade.py" > "$dir/reward.json"
