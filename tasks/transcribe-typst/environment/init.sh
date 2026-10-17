#!/bin/sh
# Reset phase: image build layers carry stale mtimes; make the recordings look
# recently recorded.
touch /root/notes/*.mp3 2>/dev/null || true
