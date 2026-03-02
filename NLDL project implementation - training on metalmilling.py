from archisound import ArchiSound
from ts2vec import TS2Vec
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import NLDLarchitectures
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set directory to the project folder
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

sampling_rate = 51200 # Sampling rate for the accelerometer signals
segment_len = 10000 
stride = segment_len // 2

# Import sound encoder
autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")

# Import trained accelerometer encoder for Metal milling anomaly that we trained earlier
acc_model = TS2Vec(input_dims=2, device=device, output_dims=64)
acc_model.load("TS2Vec_metalmillinganomaly_trained.pth")

# Define lightweight neural nets and autoencoders, use 4x compression for autoencoders
M12 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
M21 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
AE1 = NLDLarchitectures.AutoEncoder(64, 64, 16)
AE2 = NLDLarchitectures.AutoEncoder(64, 64, 16)
AE3 = NLDLarchitectures.AutoEncoder(128, 128, 32)

# Extract non-anomalous Metal milling anomaly data (disjoint from the training data for TS2Vec) and apply decoders to each modality 
file_path = "Metal milling anomaly/imi_vm20i/dataset11/20240226_061422.328211_acc.csv"
file_path_labels = "Metal milling anomaly/imi_vm20i/dataset11/label.csv"

acc_data = np.loadtxt(file_path, delimiter=",", dtype=np.float32, skiprows=1)
acc_labels = np.loadtxt(file_path_labels, delimiter=",", dtype=str, skiprows=1)
acc_labels = acc_labels[:,0:2].astype(float)
print(acc_data.shape)

# Compute starting and stopping indices for the cutting signals 
starting_times = acc_labels[:,0] * sampling_rate
starting_times = starting_times.astype(int)
stopping_times = acc_labels[:,1] * sampling_rate
stopping_times = stopping_times.astype(int)

# Make a list of all training windows for accelerometer signals 
series_list = []

for i in range(len(starting_times)):
    j = starting_times[i]
    while j + segment_len <= stopping_times[i]:
        series_list.append(acc_data[(j-1):(j-1+segment_len),:])
        j += stride

cutting_series = np.stack(series_list, axis=0)
num_segments = len(series_list)

# Reshape our array to the correct shape, extract 2 accelerometer signals and downsample
final_acc = cutting_series.reshape(num_segments, segment_len, cutting_series.shape[2])[:,::10,1:3]

mean_cut = final_acc.mean(axis=(0, 1), keepdims=True)  # shape (1, 1, 2)
std_cut  = final_acc.std(axis=(0, 1), keepdims=True)   # shape (1, 1, 2)

# Standardize each channel to mean 0 and variance 1
final_acc = (final_acc - mean_cut) / std_cut

# Now, extract non-anomalous sound signals 
file_path_sound = "Metal milling anomaly/imi_vm20i/dataset11/20240226_061422.328211_s0.wav"
sr, audio = wavfile.read(file_path_sound)

# Audio is sampled at 48kHz (not 51.2kHz), so we need to compute the times and stride for audio as well
audio_starts = acc_labels[:,0] * sr
audio_ends = acc_labels[:,1] * sr
audio_starts = audio_starts.astype(int)
audio_ends = audio_ends.astype(int)
audio_stride = math.ceil(stride * sr / sampling_rate)

sample_time = segment_len/sampling_rate
audio_length = int(sample_time * sr) # 10000 steps in accelerometer domain corresponds to 9375 in audio domain

audio_series_list = []

for i in range(len(acc_labels[:,0])):
    j = audio_starts[i]
    while j + audio_length <= audio_ends[i]:
        audio_series_list.append(audio[(j-1):(j-1+audio_length)])
        j += audio_stride

cutting_audio = np.stack(audio_series_list, axis=0)
final_audio = cutting_audio.reshape(num_segments, audio_length, 1)[:,::10,:]

mean_audio = final_audio.mean(axis=(0,1), keepdims=True)
std_audio = final_audio.std(axis=(0,1), keepdims=True)

print(mean_audio)
print(std_audio)

# Standardize audio data to mean 0 and variance 1
final_audio = (final_audio - mean_audio) / std_audio
final_audio = np.transpose(np.concatenate([final_audio, final_audio], axis=-1), (0, 2, 1))
final_audio = torch.tensor(final_audio, dtype=torch.float32)

# We extract the features for training the cross-modal maps
patch_size = 4 
remainder = (audio_length // 10) % patch_size 
pad = patch_size - remainder

acc_features = acc_model.encode(final_acc).max(axis=1) # Max pooling over timesteps 
acc_features = torch.tensor(acc_features, dtype=torch.float32)

final_audio = nn.functional.pad(final_audio, (0, pad)) # Sound encoder only takes inputs lengths that are multiples of 4, therefore we pad with 0s
sound_features = autoencoder.encode(final_audio).reshape(num_segments, 64).detach() # Stack the vector representations to achieve a 64-dimensional vector

# Training loop for cross-modal maps 
batch_size = 64
epochs = 1000
lr = 0.0001
loss_fn = nn.MSELoss()
optimizer12 = optim.Adam(M12.parameters(), lr=lr)
optimizer21 = optim.Adam(M21.parameters(), lr=lr)
optimizerAE1 = optim.Adam(AE1.parameters(), lr=lr)
optimizerAE2 = optim.Adam(AE2.parameters(), lr=lr)
optimizerAE3 = optim.Adam(AE3.parameters(), lr=lr)

loss_list_12 = []
loss_list_21 = []
loss_list_AE1 = []
loss_list_AE2 = []
loss_list_AE3 = []

for i in range(epochs):
    idx = np.random.choice(num_segments, batch_size, replace=True)
    acc_batch = acc_features[idx]
    sound_batch = sound_features[idx]
    combined = torch.cat([acc_batch, sound_batch], dim=1) # Combined features for third autoencoder

    pred12 = M12(acc_batch)
    pred21 = M21(sound_batch)
    predAE1 = AE1(acc_batch)
    predAE2 = AE2(sound_batch)
    predAE3 = AE3(combined)

    loss12 = loss_fn(pred12, sound_batch)
    loss21 = loss_fn(pred21, acc_batch)
    lossAE1 = loss_fn(predAE1, acc_batch)
    lossAE2 = loss_fn(predAE2, sound_batch)
    lossAE3 = loss_fn(predAE3, combined)

    optimizer12.zero_grad()
    optimizer21.zero_grad()
    optimizerAE1.zero_grad()
    optimizerAE2.zero_grad()
    optimizerAE3.zero_grad()

    loss12.backward()
    loss21.backward()
    lossAE1.backward()
    lossAE2.backward()
    lossAE3.backward()

    optimizer12.step()
    optimizer21.step()
    optimizerAE1.step()
    optimizerAE2.step()
    optimizerAE3.step()

    if i % 10 == 0:
        loss_list_12.append(loss12.item())
        loss_list_21.append(loss21.item())
        loss_list_AE1.append(lossAE1.item())
        loss_list_AE2.append(lossAE2.item())
        loss_list_AE3.append(lossAE3.item())

# Save the cross-modal models for inference later 
save_file = "Crossmodal_metalmillinganomaly_trained.pth"
torch.save({"M12": M12.state_dict(),
            "M21": M21.state_dict(),
            "AE1": AE1.state_dict(),
            "AE2": AE2.state_dict(),
            "AE3": AE3.state_dict()
            }, save_file)

# Plot training losses to show convergence
plt.plot(list(range(0, 1000, 10)), loss_list_12)
plt.title("Training loss for M12")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.grid(True)
plt.show()

plt.plot(list(range(0, 1000, 10)), loss_list_21)
plt.title("Training loss for M21")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.grid(True)
plt.show()

plt.plot(list(range(0, 1000, 10)), loss_list_AE1)
plt.title("Training loss for AE1")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.grid(True)
plt.show()

plt.plot(list(range(0, 1000, 10)), loss_list_AE2)
plt.title("Training loss for AE2")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.grid(True)
plt.show()

plt.plot(list(range(0, 1000, 10)), loss_list_AE3)
plt.title("Training loss for AE3")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.grid(True)
plt.show()
