#!/usr/bin/env bash
# Wait until the shared machine allows a multi-threaded measurement:
# available memory >= 8 GB and 1-minute load average < 4.
while true; do
    avail=$(free -g | awk '/^Mem:/ {print $7}')
    load=$(awk '{print $1}' /proc/loadavg)
    if [ "$avail" -ge 8 ] && awk -v l="$load" 'BEGIN {exit !(l < 4)}'; then
        echo "gate: available=${avail}G load=${load}"
        exit 0
    fi
    sleep 30
done
