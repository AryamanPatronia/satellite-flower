# #!/bin/bash

# # Wait for the container to start...
# MAX_WAIT=30
# TIME_WAITED=0

# echo "[+] Waiting for status_dashboard container to start..."

# until docker ps --filter "name=status_dashboard" --filter "status=running" | grep status_dashboard &> /dev/null
# do
#     sleep 1
#     TIME_WAITED=$((TIME_WAITED + 1))
#     if [ $TIME_WAITED -ge $MAX_WAIT ]; then
#         echo "Error: status_dashboard container did not start within $MAX_WAIT seconds."
#         exit 1
#     fi
# done

# echo "[+] Container is running. Launching dashboard window..."

# osascript <<EOF &
# tell application "Terminal"
#     do script "cd \"$(pwd)\" && docker exec -it status_dashboard python status_monitor.py"
# end tell
# EOF
