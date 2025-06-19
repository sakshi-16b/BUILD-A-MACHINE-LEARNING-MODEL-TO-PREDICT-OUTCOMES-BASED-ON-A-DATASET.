import os
import pandas as pd
import numpy as np
import time
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt

# --- Configuration ---
OUTPUT_CSV_FILE = "light_transaction_data.csv"
NUM_ROWS = 500_000
DASK_N_WORKERS = 2
DASK_THREADS = 1
DASK_MEMORY = '1GB'

def generate_csv(filename, rows):
    print(f"Generating {rows} rows...")
    products = ["Laptop", "Monitor", "Keyboard", "Mouse", "Webcam"]
    regions = ["North", "South", "East", "West"]
    customers = [f"C_{i}" for i in range(100)]

    chunk_size = 50_000
    with open(filename, 'w') as f:
        f.write("TransactionID,CustomerID,ProductID,Region,Quantity,UnitPrice,TransactionTimestamp\n")
        for i in range(0, rows, chunk_size):
            this_chunk = min(chunk_size, rows - i)
            df = pd.DataFrame({
                "TransactionID": np.arange(i + 1, i + this_chunk + 1),
                "CustomerID": np.random.choice(customers, this_chunk),
                "ProductID": np.random.choice(products, this_chunk),
                "Region": np.random.choice(regions, this_chunk),
                "Quantity": np.random.randint(1, 5, this_chunk),
                "UnitPrice": np.round(np.random.uniform(100, 300, this_chunk), 2),
                "TransactionTimestamp": pd.to_datetime("2023-01-01") +
                    pd.to_timedelta(np.random.randint(0, 30 * 24 * 60 * 60, this_chunk), unit='s')
            })
            df.to_csv(f, index=False, header=False)

# --- Main ---
if __name__ == '__main__':
    cluster = LocalCluster(
        n_workers=DASK_N_WORKERS,
        threads_per_worker=DASK_THREADS,
        memory_limit=DASK_MEMORY,
        processes=True
    )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    if not os.path.exists(OUTPUT_CSV_FILE):
        generate_csv(OUTPUT_CSV_FILE, NUM_ROWS)

    print("Reading with Dask...")
    ddf = dd.read_csv(OUTPUT_CSV_FILE,
                      parse_dates=['TransactionTimestamp'],
                      blocksize='16MB')

    print("Partitions:", ddf.npartitions)

    print("Processing grouped average...")
    ddf['Month'] = ddf['TransactionTimestamp'].dt.month
    grouped = ddf.groupby(['ProductID', 'Region', 'Month'])['Quantity'].mean()
    result = grouped.compute().reset_index()

    print("Top 5 results:\n", result.head())

    print("Generating bar chart...")
    pivot = result.pivot_table(index='ProductID', columns='Region', values='Quantity', aggfunc='mean')
    pivot.plot(kind='bar', figsize=(10, 6))
    plt.title("Avg Quantity per Product by Region")
    plt.ylabel("Avg Quantity Sold")
    plt.tight_layout()
    plt.show()

    client.close()
    cluster.close()
    print("Finished and cleaned up.")
