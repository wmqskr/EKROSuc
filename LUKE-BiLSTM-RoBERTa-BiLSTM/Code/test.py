import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LukeForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef, f1_score
from torch.utils.data import TensorDataset
from LUKEClassifier import *
from RoBERTaClassifier import *




# Set random seed
torch.manual_seed(42)
np.random.seed(42)

  
# Extract sequences
def remove_name(data):
    data_new = []
    for i in range(1,len(data),2):
        data_new.append(data[i])
    return data_new
 

# Read data
# with  open("../dataset/independenttestdataset.fasta") as f:
with  open("../dataset/rmX_49.fasta") as f:
    pos_neg_Data= f.readlines()
    pos_neg_Data = [s.strip() for s in pos_neg_Data]
print(len(pos_neg_Data))
print("Data reading completed")
print("———————————————————————————————————————————————————")

pos_neg_Data = remove_name(pos_neg_Data)

print(len(pos_neg_Data),len(pos_neg_Data[0]))
print("Sequence extraction completed")
print("———————————————————————————————————————————————————")


# Define labels
# pos_neg_label = np.concatenate([np.ones(1010), np.zeros(1010)], axis=0)  # Vertical concatenation
pos_neg_label = np.concatenate([np.ones(963), np.zeros(1007)], axis=0)  # Vertical concatenation
print(pos_neg_label.shape)
print("Label definition completed")
print("———————————————————————————————————————————————————")