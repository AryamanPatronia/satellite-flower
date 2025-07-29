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

# Run data download and partition
echo "Downloading EuroSAT dataset..."
python3 download_eurosat.py


python3 partition_eurosat.py

echo "Printing dataset stats for each client..."
python3 VerifyEuroSAT.py

echo "Setup complete."
echo "Make sure docker is running in the background."
echo "You can now run the simulation using:"
echo "    ./start_all.sh"
echo ""

