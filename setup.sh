echo "[+] Setting up local virtual environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment and dependencies installed."

# Cleanup...
echo "Cleaning data folder (if exists)..."
rm -rf data/clients_data
rm -rf data/raw

# Install Pumba for chaos simulation(docker should be installed)...
echo "Installing Pumba for chaos simulation..."
echo "Pulling Pumba Docker image for chaos simulation (required for injecting chaos)..."
docker pull gaiaadm/pumba:latest

# Run data download and partition
echo "Downloading EuroSAT dataset..."
python3 download_eurosat.py

# Centralize the test set...
echo "Centralizing test set..."
python3 central_testset.py

# Partition the dataset for clients...
echo "Partitioning EuroSAT dataset for clients..."
python3 partition_eurosat.py

# Create the constellation files...
echo "Creating constellation files..."
python3 constellation_maker.py

#Create cluster maps...
echo "Creating cluster maps..."
python3 generate_cluster_maps.py


# Verify the setup...
echo "Checking if the central test set is created..."
python3 VerifyCentral.py

# Check if and how the client datasets are created...
echo "Checking if the client datasets are created..."
python3 VerifyEuroSAT.py

echo "Setup complete."
echo "Make sure docker is running in the background."

echo "You can now try ./start_FedAsync.sh or ./start_FedAvg.sh to start Federated Learning in Satellite Constellations..."
