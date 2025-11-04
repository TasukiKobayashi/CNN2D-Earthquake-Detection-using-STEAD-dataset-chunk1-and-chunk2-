#import library
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import time
import gc

matplotlib.use("Agg")

#file paths
dst1_hdf5 = "/Users/whynot.son/Downloads/Dataset/chunk1.hdf5"
dst1_csv  = "/Users/whynot.son/Downloads/Dataset/chunk1.csv"
dst2_hdf5 = "/Users/whynot.son/Downloads/Dataset/chunk2.hdf5"
dst2_csv  = "/Users/whynot.son/Downloads/Dataset/chunk2.csv"

img_save_path = "/Users/whynot.son/Downloads/Test/image"
os.makedirs(img_save_path, exist_ok=True)

#worker function (spectrogram)
def process_one_trace(args):
    trace_name, hdf5_path, img_save_path = args
    try:
        with h5py.File(hdf5_path, "r") as dtfl:
            if trace_name not in dtfl["data"]:
                return (trace_name, 0, "not found")
            data = np.array(dtfl["data"][trace_name])

            #spectrogram
            fig, ax = plt.subplots(figsize=(3,2))
            ax.specgram(data[:, 2], Fs=100, NFFT=1024, cmap="gray", vmin=-10, vmax=25) #upscale to 1024, 2048 but cost a lot of time in NFFT
            ax.set_xlim([0, 60])
            ax.axis("off")

            spec_path = f"{img_save_path}/{trace_name}.png"
            plt.savefig(spec_path, bbox_inches="tight", transparent=True, pad_inches=0, dpi=50)
            size = os.path.getsize(spec_path)
            plt.close()

            return (trace_name, size, None)

    except Exception as e:
        return (trace_name, 0, str(e))

#batch processor
def process_chunk(hdf5_path, traces, batch_size=1000):
    total_size = 0
    num_batches = (len(traces) + batch_size - 1) // batch_size

    for i in range(0, len(traces), batch_size):
        batch = traces[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size+1}/{num_batches} ({len(batch)} traces)")

        args = [(t, hdf5_path, img_save_path) for t in batch]
        failed_traces = []

        with Pool(processes=min(cpu_count(), 8)) as pool:  #14 workers on Bao Long computer
            results_iter = pool.imap(process_one_trace, args, chunksize=10)

            results = []
            for r in tqdm(results_iter,
                          total=len(batch),
                          desc=f"{os.path.basename(hdf5_path)} batch {i//batch_size+1}"):
                results.append(r)

        batch_size_bytes = 0
        for trace, size, error in results:
            batch_size_bytes += size
            if size == 0:
                failed_traces.append(trace)

        total_size += batch_size_bytes
        print(f"Batch {i//batch_size+1} done, saved {batch_size_bytes/1e6:.2f} MB")
        if failed_traces:
            print(f"Failed {len(failed_traces)} traces: {failed_traces[:10]}{' ...' if len(failed_traces) > 10 else ''}")

        del results, batch, args, failed_traces
        gc.collect()
        time.sleep(1)

    return total_size

#main
if __name__ == "__main__":
    set_start_method("spawn", force=True)

    #read CSV
    chunk1_csv = pd.read_csv(dst1_csv, low_memory=False)
    chunk2_csv = pd.read_csv(dst2_csv, low_memory=False)

    traces1 = chunk1_csv["trace_name"].to_list()  #noise
    traces2 = chunk2_csv["trace_name"].to_list()  #earthquake

    print("Chunk1 traces:", len(traces1))
    print("Chunk2 traces:", len(traces2))

    # run processing
    size1 = process_chunk(dst1_hdf5, traces1, batch_size=5000)
    size2 = process_chunk(dst2_hdf5, traces2, batch_size=5000)
    print("All data saved:", (size1+size2)/1e9, "GB")