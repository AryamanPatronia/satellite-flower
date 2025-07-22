echo "[+] Starting full federated learning system..."

cd docker

#Remove existing containers and images to have a fresh start...
echo "[+] Cleaning up existing containers and images..."
docker-compose down -v --remove-orphans
docker image prune -a -f

# Build and start containers
echo "[+] Building and starting Docker containers..."
docker-compose up --build -d
cd ..

echo "[+] Waiting for containers to stabilize..."
sleep 5

echo "[+] Launching live dashboard..."
bash dashboard/launch_dashboard.sh

# Start using ./start_all.sh 
