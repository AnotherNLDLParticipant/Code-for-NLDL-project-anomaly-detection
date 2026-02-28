from archisound import ArchiSound
from ts2vec import TS2Vec
import os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import NLDLarchitectures
import NLDLanomalyscores
import NLDLplotfunction
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sklearn.metrics as sk

with torch.no_grad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set directory to the project folder
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    sampling_rate = 51200
    segment_len = int(10000)

    # Load test data - we take 30 normal and 30 anomalous cutting signals
    acc1 = "Metal milling anomaly/imi_vm20i/dataset1/20240229_053831.227000_acc.csv" # pure anomalies of type 1
    acc2 = "Metal milling anomaly/imi_vm20i/dataset2/20240229_055516.387000_acc.csv" # pure anomalies of type 2
    acc12 = "Metal milling anomaly/imi_vm20i/dataset12/20240226_062848.426000_acc.csv"  # pure normal data

    sound1 = "Metal milling anomaly/imi_vm20i/dataset1/20240229_053831.227000_s0.wav"
    sound2 = "Metal milling anomaly/imi_vm20i/dataset2/20240229_055516.387000_s0.wav"
    sound12 = "Metal milling anomaly/imi_vm20i/dataset12/20240226_062848.426000_s0.wav"

    labels1 = np.loadtxt("Metal milling anomaly/imi_vm20i/dataset1/label.csv", delimiter=",", dtype=str, skiprows=1)[:,0:2].astype(float)
    labels2 = np.loadtxt("Metal milling anomaly/imi_vm20i/dataset2/label.csv", delimiter=",", dtype=str, skiprows=1)[:,0:2].astype(float)
    labels12 = np.loadtxt("Metal milling anomaly/imi_vm20i/dataset12/label.csv", delimiter=",", dtype=str, skiprows=1)[:30,0:2].astype(float)

    end12acc = int(labels12[29,1] * sampling_rate)

    acc1_np = np.loadtxt(acc1, delimiter=",", dtype=np.float32, skiprows=1)[:,1:3]
    acc2_np = np.loadtxt(acc2, delimiter=",", dtype=np.float32, skiprows=1)[:,1:3]
    acc12_np = np.loadtxt(acc12, delimiter=",", dtype=np.float32, skiprows=1)[0:end12acc,1:3]

    sr, sound1_np = wavfile.read(sound1)
    sr, sound2_np = wavfile.read(sound2)
    sr, sound12_np = wavfile.read(sound12)
    
    sound_rate = sr
    audio_len = int(segment_len/sampling_rate*sound_rate) # 9375

    # Create arrays with starting and stopping times for cutting signals
    starting_times_array = np.zeros((60, 2))
    stopping_times_array = np.zeros((60, 2))

    upper1 = int(len(labels1))
    upper2 = int(len(labels1) + len(labels2))
    upper3 = int(len(labels1) + len(labels2) + len(labels12))

    starting_times_array[:upper1,0] = (labels1[:,0] * sampling_rate)
    starting_times_array[:upper1,1] = (labels1[:,0] * sound_rate)

    stopping_times_array[:upper1,0] = (labels1[:,1] * sampling_rate)
    stopping_times_array[:upper1,1] = (labels1[:,1] * sound_rate)

    starting_times_array[upper1:upper2,0] = (labels2[:,0] * sampling_rate)
    starting_times_array[upper1:upper2,1] = (labels2[:,0] * sound_rate)

    stopping_times_array[upper1:upper2,0] = (labels2[:,1] * sampling_rate)
    stopping_times_array[upper1:upper2,1] = (labels2[:,1] * sound_rate)

    starting_times_array[upper2:upper3,0] = (labels12[:,0] * sampling_rate)
    starting_times_array[upper2:upper3,1] = (labels12[:,0] * sound_rate)

    stopping_times_array[upper2:upper3,0] = (labels12[:,1] * sampling_rate)
    stopping_times_array[upper2:upper3,1] = (labels12[:,1] * sound_rate)

    starting_times_array = starting_times_array.astype(int)
    stopping_times_array = stopping_times_array.astype(int)

    acc_list = []
    sound_list = []
    label_list = []

    # Fill up acc_list with windows
    for i in range(60):
        j_acc = starting_times_array[i, 0]

        while j_acc + segment_len <= stopping_times_array[i, 0]:
            if i <= upper1:
                acc_list.append(acc1_np[(j_acc - 1):(j_acc - 1 + segment_len),:])
                label_list.append(1)

            elif i > upper1 and i <= upper2:
                acc_list.append(acc2_np[(j_acc - 1):(j_acc - 1 + segment_len),:])
                label_list.append(1)
            
            else: 
                acc_list.append(acc12_np[(j_acc - 1):(j_acc - 1 + segment_len),:])
                label_list.append(0)
            
            j_acc += segment_len
    
    num_segments = len(acc_list)
    cutting_series = np.stack(acc_list, axis=0)
    label_list = np.array(label_list)

    # Fill up sound list with windows
    for i in range(60):
        j_sound = starting_times_array[i, 1]

        while j_sound + audio_len <= stopping_times_array[i, 1]:
            if i <= upper1:
                sound_list.append(sound1_np[(j_sound - 1):(j_sound - 1 + audio_len)])

            elif i > upper1 and i <= upper2:
                sound_list.append(sound2_np[(j_sound - 1):(j_sound - 1 + audio_len)])

            else: 
                sound_list.append(sound12_np[(j_sound - 1):(j_sound - 1 + audio_len)])

            j_sound += audio_len

    audio_series = np.stack(sound_list, axis=0)
    print(len(acc_list), len(sound_list), len(label_list))

    cutting_series = cutting_series.reshape(num_segments, segment_len, cutting_series.shape[2])[:,::10,:]
    audio_series = audio_series.reshape(num_segments, audio_len, 1)[:,::10,:]
    
    # Mean and standard deviation computed from normal data in the training file 
    cutting_series_mean = np.array([[[11.276512, 42.782585]]])
    cutting_series_std = np.array([[[170.04518, 112.411934]]])
    audio_series_mean = np.array([[[10.03982812]]])
    audio_series_std = np.array([[[11755.45308855]]])

    cutting_series = (cutting_series - cutting_series_mean)/cutting_series_std
    audio_series = (audio_series - audio_series_mean)/audio_series_std

    audio_series  = np.transpose(np.concatenate([audio_series, audio_series], axis=-1), (0, 2, 1))
    audio_series = torch.tensor(audio_series, dtype=torch.float32)
    
    # Import sound encoder 
    autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")

    # Import accelerometer encoder for Metal milling anomaly that we trained earlier 
    acc_model = TS2Vec(input_dims=2, device=device, output_dims=64)
    acc_model.load("TS2Vec_metalmillinganomaly_trained.pth")

    # Import cross-modal maps that we trained earlier 
    M12 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
    M21 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
    AE1 = NLDLarchitectures.AutoEncoder(64, 64, 16)
    AE2 = NLDLarchitectures.AutoEncoder(64, 64, 16)
    AE3 = NLDLarchitectures.AutoEncoder(128, 128, 32)

    modelweights = torch.load("Crossmodal_metalmillinganomaly_trained.pth", map_location=device)

    M12.load_state_dict(modelweights["M12"])
    M21.load_state_dict(modelweights["M21"])
    AE1.load_state_dict(modelweights["AE1"])
    AE2.load_state_dict(modelweights["AE2"])
    AE3.load_state_dict(modelweights["AE3"])

    M12.eval()
    M21.eval()
    AE1.eval()
    AE2.eval()
    AE3.eval()

    # Feature extractions 
    patch_size = 4 
    remainder = 938 % patch_size
    pad = patch_size - remainder

    acc_features = acc_model.encode(cutting_series).max(axis=1) # Max pooling as in TS2Vec paper 
    acc_features = torch.tensor(acc_features, dtype=torch.float32)

    audio_series = nn.functional.pad(audio_series, (0, pad)) # Pad to make time series length a multiple of 4
    sound_features = autoencoder.encode(audio_series).reshape(num_segments, 64).detach() 

    print(autoencoder.encode(audio_series).shape)

    # Cross-modal mapping 
    F2hat = M12(acc_features)
    F1hat = M21(sound_features)

    scores = NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Euclidean', aggregation='Sum')

    AE1pred = AE1(acc_features)
    AE2pred = AE2(sound_features)
    AE3pred = AE3(torch.cat([acc_features, sound_features], dim=1))
 
    # Compute reconstruction error for autoencoders
    AE1scores = torch.norm(AE1pred - acc_features, dim=1).numpy()
    AE2scores = torch.norm(AE2pred - sound_features, dim=1).numpy()
    AE3scores = torch.norm(AE3pred - torch.cat([acc_features, sound_features], dim=1), dim=1).numpy()

    # Plot histogram of error distributions and print metrics of interest
    print(NLDLplotfunction.ScoreHistogram(label_list, scores))

    auroc = sk.roc_auc_score(label_list, scores)

    fpr, tpr, thresholds = sk.roc_curve(label_list, scores)
    fprAE1, tprAE1, thresholdsAE1 = sk.roc_curve(label_list, AE1scores)
    fprAE2, tprAE2, thresholdsAE2 = sk.roc_curve(label_list, AE2scores)
    fprAE3, tprAE3, thresholdsAE3 = sk.roc_curve(label_list, AE3scores)

    # Plot all methods in the ROC below 
    plt.figure()
    plt.plot(fpr, tpr, label="Cross-modal", color="red")
    plt.plot(fprAE1, tprAE1, label="AE1", color='blue')
    plt.plot(fprAE2, tprAE2, label="AE2", color='green')
    plt.plot(fprAE3, tprAE3, label="AE3", color='orange')
    plt.plot([0,1], [0,1], "k--", label="Random guessing")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    precision, recall, thresholds = sk.precision_recall_curve(label_list, scores)
    precisionAE1, recallAE1, thresholdsAE1 = sk.precision_recall_curve(label_list, AE1scores)
    precisionAE2, recallAE2, thresholdsAE2 = sk.precision_recall_curve(label_list, AE2scores)
    precisionAE3, recallAE3, thresholdsAE3 = sk.precision_recall_curve(label_list, AE3scores)
    anomaly_prop = sum(label_list == 1)/len(label_list)

    plt.figure()
    plt.plot(recall, precision, label="Cross-modal", color="red")
    plt.plot(recallAE1, precisionAE1, label="AE1", color="blue")
    plt.plot(recallAE2, precisionAE2, label="AE2", color="green")
    plt.plot(recallAE3, precisionAE3, label="AE3", color="orange")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.axhline(y=anomaly_prop, linestyle="--", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
