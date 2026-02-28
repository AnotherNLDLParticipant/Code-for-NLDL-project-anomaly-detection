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
import NLDLanomalyscores
import numpy as np 
import math
import NLDLplotfunction
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
import sklearn.metrics as sk

with torch.no_grad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    script_dir = Path(__file__).resolve().parent

    # Change the current working directory to project folder
    os.chdir(script_dir)

    sampling_rate = 50000
    segment_len = 10000
    signal_len = 250000

    # Specify file paths to test data - 1 folder with normal data and 5 folders with different anomaly types
    horizontal = "mafaulda/horizontal-misalignment/0.5mm"
    imbalance = "mafaulda/imbalance/6g"
    overhang = "mafaulda/overhang/ball_fault/6g"
    underhang = "mafaulda/underhang/ball_fault/6g"
    vertical = "mafaulda/vertical-misalignment/0.51mm"
    normal = "mafaulda/normal"

    horizon_files = sorted(glob.glob(os.path.join(horizontal, "*.csv")))
    imbalance_files = sorted(glob.glob(os.path.join(imbalance, "*.csv")))
    overhang_files = sorted(glob.glob(os.path.join(overhang, "*.csv")))
    underhang_files = sorted(glob.glob(os.path.join(underhang, "*.csv")))
    vertical_files = sorted(glob.glob(os.path.join(vertical, "*.csv")))
    normal_files = sorted(glob.glob(os.path.join(normal, "*.csv")))[45:]

    label_list = []

    # Take the first 4 signals in each folder 
    collected_folders = [horizon_files[:4], imbalance_files[:4], overhang_files[:4], underhang_files[:4], vertical_files[:4], normal_files]
    arrays = []

    # Extract the windows for inference
    for folder in collected_folders:
        for file in folder: 
            new_data = np.genfromtxt(file, delimiter=",", skip_header=0)
            new_data = new_data.astype(np.float32)
            upper = 0
            while upper + segment_len <= signal_len:
                upper += segment_len
                arrays.append(new_data[(upper - segment_len):upper])
                if folder is collected_folders[-1]:
                    label_list.append(0)

                else: 
                    label_list.append(1)

    # Extract the 3 accelerometer signals and the sound signal 
    acc_data = np.vstack(arrays)[:,1:4]
    sound_data = np.vstack(arrays)[:,7]
    label_list = np.array(label_list)

    N_series = len(label_list)
    print(N_series)

    # Reshaping and downsampling for tractability
    acc_data = acc_data.reshape(N_series, segment_len, 3)[:,::10,:]
    sound_data = sound_data.reshape(N_series, segment_len, 1)[:,::10,:]

    # Standardization using mean and standard deviations computed earlier on normal
    acc_mean = np.array([[[ 0.0063284, -0.00064534, 0.00084082]]])
    acc_std = np.array([[[0.67140985, 0.48317337, 0.23672691]]])
    sound_mean = np.array([[[0.01176098]]])
    sound_std = np.array([[[0.19866556]]])

    acc_data = (acc_data - acc_mean) / acc_std
    sound_data = (sound_data - sound_mean) / sound_std

    # Final preparation of sound data
    sound_data = np.transpose(np.concatenate([sound_data, sound_data], axis=-1), (0, 2, 1))
    sound_data = torch.tensor(sound_data, dtype=torch.float32)

    # Load the two encoders and the model architectures
    acc_model = TS2Vec(input_dims=3, device=device, output_dims=64)
    acc_model.load("TS2Vec_mafaulda_trained.pth")

    autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")

    # Import cross-modal maps and autoencoders that we trained earlier
    M12 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
    M21 = NLDLarchitectures.Lightweight_NN(64, 64, 64)
    AE1 = NLDLarchitectures.AutoEncoder(64, 64, 16)
    AE2 = NLDLarchitectures.AutoEncoder(64, 64, 16)
    AE3 = NLDLarchitectures.AutoEncoder(128, 128, 32)

    modelweights = torch.load("Crossmodal_mafaulda_trained.pth", map_location=device)

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
    acc_features = acc_model.encode(acc_data).max(axis=1)
    acc_features = torch.tensor(acc_features, dtype=torch.float32)

    sound_features = autoencoder.encode(sound_data).reshape(N_series, 64).detach()

    # Cross modal mappings 
    F2hat = M12(acc_features)
    F1hat = M21(sound_features)

    scores = NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Euclidean', aggregation='Max')

    AE1pred = AE1(acc_features)
    AE2pred = AE2(sound_features)
    AE3pred = AE3(torch.cat([acc_features, sound_features], dim=1))

    AE1scores = torch.norm(AE1pred - acc_features, dim=1).square()
    AE2scores = torch.norm(AE2pred - sound_features, dim=1).square()
    AE3scores = torch.norm(AE3pred - torch.cat([acc_features, sound_features], dim=1), dim=1).square()

    print(NLDLplotfunction.RelevantCurves(label_list, scores))
    NLDLplotfunction.ScoreHistogram(label_list, scores)
    print("E Sum",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Euclidean', aggregation='Sum')))
    print("E prod",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Euclidean', aggregation='Product')))
    print("E 1",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Euclidean', aggregation='First')))
    print("E 2",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Euclidean', aggregation='Second')))
    print("C Max",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Cosine', aggregation='Max')))
    print("C Sum",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Cosine', aggregation='Sum')))
    print("C Prod",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Cosine', aggregation='Product')))
    print("C 1",NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Cosine', aggregation='First')))
    print("C 2", NLDLplotfunction.RelevantCurves(label_list, NLDLanomalyscores.anomaly_score(acc_features, sound_features, F1hat, F2hat, psi='Cosine', aggregation='Second')))


    print(NLDLplotfunction.RelevantCurves(label_list, AE1scores))
    print(NLDLplotfunction.RelevantCurves(label_list, AE2scores))
    print(NLDLplotfunction.RelevantCurves(label_list, AE3scores))

    fpr, tpr, thresholds = sk.roc_curve(label_list, scores)
    auroc = sk.roc_auc_score(label_list, scores)
    fprAE1, tprAE1, thresholdsAE1 = sk.roc_curve(label_list, AE1scores)
    fprAE2, tprAE2, thresholdsAE2 = sk.roc_curve(label_list, AE2scores)
    fprAE3, tprAE3, thresholdsAE3 = sk.roc_curve(label_list, AE3scores)

    anomaly_prop = sum(label_list == 1)/len(label_list)

    # Plot all methods in the ROC below 
    plt.figure()
    plt.plot(fpr, tpr, label="Cross-modal", color="red")
    plt.plot(fprAE1, tprAE1, label="AE1", color='blue')
    plt.plot(fprAE2, tprAE2, label="AE2", color='green')
    plt.plot(fprAE3, tprAE3, label="AE3", color='orange')
    plt.plot([0,1], [0,1], "k--", label="Random guessing")
    plt.xlim([0, 1])
    plt.ylim([0.9, 1.05])
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

    plt.figure()
    plt.plot(recall, precision, label="Cross-modal", color="red")
    plt.plot(recallAE1, precisionAE1, label="AE1", color="blue")
    plt.plot(recallAE2, precisionAE2, label="AE2", color="green")
    plt.plot(recallAE3, precisionAE3, label="AE3", color="orange")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.axhline(y=anomaly_prop, linestyle="--", linewidth=2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()