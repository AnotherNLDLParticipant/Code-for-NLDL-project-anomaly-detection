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
import numpy as np 
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_dir = Path(__file__).resolve().parent

# Change the current working directory to project folder
os.chdir(script_dir)

sampling_rate = 50000
segment_len = 10000
signal_len = 250000
stride = segment_len // 2

# Specify file path to training data for cross-modal
file_path = "mafaulda/normal"

# Extract csv files 26 to 45 in the above folder 
csv_files = sorted(glob.glob(os.path.join(file_path, "*.csv")))
arrays = []

for file in csv_files[25:45]:
    new_data = np.genfromtxt(file, delimiter=",", skip_header=0)
    new_data = new_data.astype(np.float32)
    upper = segment_len - stride
    while upper + stride <= signal_len:
        upper += stride
        arrays.append(new_data[(upper - segment_len):upper])

# Extract the 3 accelerometer signals and the sound signal 
acc_data = np.vstack(arrays)[:,1:4]
sound_data = np.vstack(arrays)[:,7]

# Reshaping and downsampling
N_series = int(((signal_len - segment_len)/stride + 1)*20)
acc_data = acc_data.reshape(N_series, segment_len, 3)[:,::10,:]
sound_data = sound_data.reshape(N_series, segment_len, 1)[:,::10,:]

# Standardization
acc_mean = acc_data.mean(axis=(0,1), keepdims=True)
acc_std = acc_data.std(axis=(0,1), keepdims=True)
print(acc_mean)
print(acc_std)

sound_mean = sound_data.mean(axis=(0,1), keepdims=True)
sound_std = sound_data.std(axis=(0,1), keepdims=True)
print(sound_mean)
print(sound_std)

acc_data = (acc_data - acc_mean) / acc_std
sound_data = (sound_data - sound_mean) / sound_std

# Final preparation of sound data
sound_data = np.transpose(np.concatenate([sound_data, sound_data], axis=-1), (0, 2, 1))
sound_data = torch.tensor(sound_data, dtype=torch.float32)

# Load the two encoders and the model architectures
acc_model = TS2Vec(input_dims=3, device=device, output_dims=64)
acc_model.load("TS2Vec_mafaulda_trained.pth")

autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")

M12 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
M21 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
AE1 = NLDLarchitectures.AutoEncoder(64, 64, 16)
AE2 = NLDLarchitectures.AutoEncoder(64, 64, 16)
AE3 = NLDLarchitectures.AutoEncoder(128, 128, 32)

# Feature extraction for training the cross-modal maps and the autoencoder
acc_features = acc_model.encode(acc_data).max(axis=1) # Max pooling as in TS2Vec paper
acc_features = torch.tensor(acc_features, dtype=torch.float32)

audio_features = autoencoder.encode(sound_data).reshape(N_series, 64).detach()

# Training loop for cross-modal maps and autoencoder
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
    idx = np.random.choice(N_series, batch_size, replace=True)
    acc_batch = acc_features[idx]
    sound_batch = audio_features[idx]
    combined = torch.cat([acc_batch, sound_batch], dim=1)

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

save_file = "Crossmodal_mafaulda_trained.pth"
torch.save({"M12": M12.state_dict(),
            "M21": M21.state_dict(),
            "AE1": AE1.state_dict(),
            "AE2": AE2.state_dict(),
            "AE3": AE3.state_dict()
            }, save_file)

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


