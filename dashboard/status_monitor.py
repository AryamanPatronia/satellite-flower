from time import sleep
import os
from rich.console import Console
from rich.table import Table

SATELLITE_IDS = ["1", "2", "3", "4", "5"]
LOG_DIR = "shared_logs"

console = Console()

def read_status(cid):
    path = os.path.join(LOG_DIR, f"satellite_{cid}.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return "No status"

def display_dashboard():
    while True:
        table = Table(title="Satellite Training Status")

        table.add_column("Satellite", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")

        for cid in SATELLITE_IDS:
            status = read_status(cid)
            table.add_row(f"Satellite {cid}", status)

        console.clear()
        console.print(table)
        sleep(1)

if __name__ == "__main__":
    display_dashboard()
