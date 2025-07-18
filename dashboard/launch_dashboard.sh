#!/bin/bash

osascript <<EOF
tell application "Terminal"
    do script "cd \"$(pwd)\" && docker exec -it status_dashboard python status_monitor.py"
end tell
EOF
