#!/bin/bash

echo "[+] Starting full federated learning system..."

cd docker
docker-compose down -v --remove-orphans
docker-compose up --build -d
cd ..

echo "[+] Waiting for containers to stabilize..."
sleep 5

echo "[+] Launching live dashboard..."
bash dashboard/launch_dashboard.sh
