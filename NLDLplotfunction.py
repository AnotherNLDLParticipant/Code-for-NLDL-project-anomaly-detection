from ts2vec import TS2Vec
import os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import torch.nn.functional as F
import sklearn.metrics as sk
import matplotlib.pyplot as plt 

# Function creates ROC-curves and obtains AUROC 

def RelevantCurves(labels, scores):
    fpr, tpr, thresholds = sk.roc_curve(labels, scores)
    auroc = sk.roc_auc_score(labels, scores)
    anomaly_prop = sum(labels == 1)/len(labels)

    # Plot all methods in the ROC below 
    plt.figure()
    plt.plot(fpr, tpr, label="Cross-modal", color="red")
    plt.plot([0,1], [0,1], "k--", label="Random guessing")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Precision-Recall curves
    precision, recall, thresholds = sk.precision_recall_curve(labels, scores)
    auprc = sk.average_precision_score(labels, scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure()
    plt.plot(recall, precision, label="Cross-modal", color="red")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.axhline(y=anomaly_prop, linestyle="--", linewidth=2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return (auroc, auprc, best_f1, best_threshold)
    # Make histograms of anomaly scores under non-anomalous and anomalous data

def ScoreHistogram(labels, scores): 
    normal_scores = []
    anomalous_scores = []

    for i in range(len(scores)):
        if labels[i] == 0:
            normal_scores.append(scores[i])

        else: 
            anomalous_scores.append(scores[i])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=120, sharey=True)
    bins = 30
    
    # Histogram for normal scores
    axes[0].hist(
        normal_scores,
        bins=bins, 
        color="#4C72B0",
        edgecolor="white",
        alpha=0.8
    )

    axes[0].set_title("Normal scores")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")

    # Histogram for anomalous scores
    axes[1].hist(
        anomalous_scores,
        bins=bins,
        color = "#DD8452",
        edgecolor= "white",
        alpha=0.8
    )

    axes[1].set_title("Anomalous scores")
    axes[1].set_xlabel("Value")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


