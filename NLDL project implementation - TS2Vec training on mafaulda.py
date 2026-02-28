import datautils
from ts2vec import TS2Vec
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_dir = Path(__file__).resolve().parent

# Change the current working directory
os.chdir(script_dir)

sampling_rate = 50000 # Sampling rate is 50kHz
segment_len = 10000
signal_len = 250000
stride = segment_len // 2

file_path = "mafaulda/normal"

# Extract the first 25 csv files from the folder with normal signals
csv_files = sorted(glob.glob(os.path.join(file_path, "*.csv")))
arrays = []

for file in csv_files[:25]:
    new_data = np.genfromtxt(file, delimiter=",", skip_header=0)
    new_data = new_data.astype(np.float32)
    upper = segment_len - stride
    while upper + stride <= signal_len:
        upper += stride
        arrays.append(new_data[(upper - segment_len):upper])

# Extract the 3 accelerometer signals
X = np.vstack(arrays)[:,1:4]
print(X.shape)

# Compute number of samples
N_series = int(((signal_len - segment_len)/stride + 1)*25)
print(N_series)

# Reshape and downsample the time series for tractability
X = X.reshape(N_series, segment_len, 3)[:,::10,:]
mean = X.mean(axis=(0, 1), keepdims=True)  # shape (1, 1, 3)
std  = X.std(axis=(0, 1), keepdims=True)   # shape (1, 1, 3)

# Standardize each channel to mean 0 and variance 1
X = (X - mean) / std

# Define, fit and save the TS2Vec model
model = TS2Vec(input_dims=3, device=device, output_dims=64, batch_size=4)
model.fit(X, verbose=True, n_epochs=200)
model.save("TS2Vec_mafaulda_trained.pth")



