# Federated Learning in Satellite Constellations using Flower

## Setup Instructions

1. **Clone the repository**

   ```sh
   git clone https://github.com/AryamanPatronia/satellite-flower

   ```

2. **Installing Docker**

   - You have to make sure Docker Desktop is installed and running on your machine.  
     [Download Docker](https://www.docker.com/products/docker-desktop/)

3. **Run the Setup Script**

   - This will create a Python virtual environment, install dependencies, download the EuroSAT dataset, partition the data, and generate all the required configuration files.

   ```sh
   ./setup.sh
   ```

4. **Run a Simulation**

   - To start a federated learning simulation, use one of the following:
     - **FedAvg:**
       ```sh
       ./start_FedAvg.sh
       ```
     - **FedAsync:**
       ```sh
       ./start_FedFedAsync.sh
       ```

5. **View Results**
   - After training, the chart and the history will be saved in the results/results_FedAvg & results/results_FedAsync folder.
   - You can visualize the results of both strategies of all constellations using the provided plotting scripts, e.g.:
     ```sh
     python3 plot_all_FedAvg.py
     python3 plot_all_FedAsync.py
     python3 PlotAll.py
     ```

---

## Notes

- All data and generated files are ignored by git, so you must run [setup.sh](http://_vscodecontentref_/2) before running any simulations.
- The [constellations](http://_vscodecontentref_/3) folder will be set up automatically and you do not need to edit or add files there.
- If you encounter any issues, check if Docker is running properly and your Python virtual environment is activated.

---

## Project Structure

- [setup.sh](http://_vscodecontentref_/4) – One command setup for environment, data, and configuration.
- [start_FedAvg.sh](http://_vscodecontentref_/5) / [start_FedAsync.sh](http://_vscodecontentref_/6) – Scripts to launch the federated learning simulations.
- [plot_all_FedAvg.py](http://_vscodecontentref_/7) / `plot_all_FedAsync.py` – Scripts to visualize results.
- [constellations](http://_vscodecontentref_/8) – Contains auto-generated constellation schedules.
- [data](http://_vscodecontentref_/9) – Contains downloaded and partitioned datasets (ignored by git).
- [results](http://_vscodecontentref_/10) – Contains experiment results and logs (ignored by git).

---

For any questions, you can email [A.Patronia2@newcastle.ac.uk](mailto:A.Patronia2@newcastle.ac.uk)  
or my personal email: [aryanpatronia3@gmail.com](mailto:aryanpatronia3@gmail.com)
