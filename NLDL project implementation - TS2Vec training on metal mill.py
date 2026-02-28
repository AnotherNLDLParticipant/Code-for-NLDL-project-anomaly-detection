from ts2vec import TS2Vec
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_dir = Path(__file__).resolve().parent

sampling_rate = 51200 # Sampling rate is 51.2 kHz for the accelerometer signals
segment_len = 10000
stride = segment_len // 2

# Change the current working directory to the relevant folder
os.chdir(script_dir)

# Location of non-anomalous data to be used for training
file_path = "Metal milling anomaly/imi_vm20i/dataset10/20240226_055737.825160_acc.csv"
file_path_labels = "Metal milling anomaly/imi_vm20i/dataset10/label.csv"

acc_data = np.loadtxt(file_path, delimiter=",", dtype=np.float32, skiprows=1)
acc_labels = np.loadtxt(file_path_labels, delimiter=",", dtype=str, skiprows=1)
acc_labels = acc_labels[:,0:2].astype(float) # Index 0 and 1 are the starting and stopping times

# Convert time in seconds for cutting signals to time series index
starting_times = acc_labels[:,0] * sampling_rate
starting_times = starting_times.astype(int)
stopping_times = acc_labels[:,1] * sampling_rate
stopping_times = stopping_times.astype(int)

# Create a list of all training windows
series_list = []

for i in range(len(starting_times)):
    j = starting_times[i]
    while j + segment_len <= stopping_times[i]:
        series_list.append(acc_data[(j-1):(j-1+segment_len),:])
        j += stride

cutting_series = np.stack(series_list, axis=0)
num_segments = len(series_list)

# Pick out the 2 accelerometer signals and downsample for tractability
final_data = cutting_series.reshape(num_segments, segment_len, cutting_series.shape[2])[:,::10,1:3]
print(final_data.shape)

# Standardize each channel to have mean 0 and variance 1
mean_cut = final_data.mean(axis=(0, 1), keepdims=True)  # shape (1, 1, 2)
std_cut  = final_data.std(axis=(0, 1), keepdims=True)   # shape (1, 1, 2)
final_data = (final_data - mean_cut) / std_cut

# Define the TS2Vec model, train with contrastive learning described in project and save the model weights
model = TS2Vec(input_dims=2, device=device, output_dims=64, batch_size=4)
model.fit(final_data, verbose=True, n_epochs=200)
model.save("TS2Vec_metalmillinganomaly_trained.pth")