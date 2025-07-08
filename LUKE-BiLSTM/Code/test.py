import torch
import torch.nn as nn
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
from Classifier import *




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
with open("../dataset/independenttestdataset.fasta") as f:
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
pos_neg_label = np.concatenate([np.ones(1010), np.zeros(1010)], axis=0)  # Vertical concatenation

print(pos_neg_label.shape)
print("Label definition completed")
print("———————————————————————————————————————————————————")


def test_model(test_data, test_label):
    # Device setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("../luke-base")
    model = Classifier().to(device)
    
    # Load the best saved model
    model.load_state_dict(torch.load('../Result/lukebilstm_model.pth', map_location='cuda:0'))
    model.eval()

    # Data preprocessing
    encoded_texts = tokenizer(test_data, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(test_label, dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoader
    test_dataset = TensorDataset(input_ids, attention_mask, labels)
    test_loader = DataLoader(test_dataset, batch_size=34, shuffle=False)

    # Inference on test data
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in test_loader:
            outputs = model(batch_input_ids, batch_attention_mask, flag=1)
            batch_preds = (outputs >= 0.5).squeeze().cpu().numpy()
            batch_probs = outputs.cpu().detach().numpy()
            all_preds.extend(batch_preds.tolist())
            all_probs.extend(batch_probs.tolist())
            all_labels.extend(batch_labels.detach().cpu().numpy())

    # Calculate metrics
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    TP = TN = FP = FN = 0
    for i in range(len(all_labels)):
        if all_preds[i] == 1 and all_labels[i] == 1:
            TP += 1
        elif all_preds[i] == 0 and all_labels[i] == 0:
            TN += 1 
        elif all_preds[i] == 1 and all_labels[i] == 0:
            FP += 1
        else:
            FN += 1
    acc = (TP + TN) / (TP + TN + FP + FN)
    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    print(f"Test Sn: {sn:.4f}, Sp: {sp:.4f}, Acc: {acc:.4f}, MCC: {mcc:.4f}, f1: {f1:.4f}, AUC: {auc:.4f}")

test_model(pos_neg_Data, pos_neg_label)
