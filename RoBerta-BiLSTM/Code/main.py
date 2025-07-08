import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from train import train_and_evaluate_model_with_cv

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Extract sequences
def remove_name(data):
    data_new = []
    for i in range(1, len(data), 2):
        data_new.append(data[i])
    return data_new

# Read data
with open("../dataset/traindataset.fasta") as f:
    pos_neg_Data = f.readlines()
    pos_neg_Data = [s.strip() for s in pos_neg_Data]
print(len(pos_neg_Data))
print("Data reading completed")
print("———————————————————————————————————————————————————")

pos_neg_Data = remove_name(pos_neg_Data)

print(len(pos_neg_Data), len(pos_neg_Data[0]))
print("Sequence extraction completed")
print("———————————————————————————————————————————————————")

# Define labels
pos_neg_label = np.concatenate([np.zeros(4039), np.ones(4039)], axis=0)  # Vertical concatenation
print(pos_neg_label.shape)
print("Label definition completed")
print("———————————————————————————————————————————————————")

# 10-fold cross-validation
train_and_evaluate_model_with_cv(pos_neg_Data, pos_neg_label)
print("Model training completed")
