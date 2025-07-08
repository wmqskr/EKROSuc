import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel, T5Tokenizer, T5EncoderModel, RobertaForSequenceClassification
import torch.nn.functional as F

class RoBERTaClassifier(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.1):
        super(RoBERTaClassifier, self).__init__()
        
        # Load pre-trained RoBERTa model
        self.roberta = RobertaForSequenceClassification.from_pretrained("../roberta-base")
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(input_size=768,  # Input size: hidden size of RoBERTa
                          hidden_size=hidden_size,  # Hidden size of LSTM
                          num_layers=3,  # Number of LSTM layers
                          bidirectional=True,  # Bi-directional LSTM
                          batch_first=True)  # Input tensor format: (batch_size, seq_length, feature_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer (input size is the output size of LSTM, 2 times hidden_size because it's bidirectional)
        self.fc = nn.Linear(hidden_size*2, 1)
        
        # Sigmoid activation function (used for binary classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, flag=0):
        # Get output from RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, 
                           output_hidden_states=True, 
                           output_attentions=True)
        
        # Last hidden state of RoBERTa, shape: (batch_size, seq_len, hidden_size)
        outputs = outputs.hidden_states[12]
        # print(outputs.shape)
        
        # Pass RoBERTa's output to LSTM layer
        lstm_output, _ = self.lstm(outputs)  # lstm_output shape: (batch_size, seq_len, hidden_size * 2)
        
        # Pooling over LSTM's output (using max pooling here)
        lstm_output = lstm_output.permute(0, 2, 1)  # Transpose, output shape: (batch_size, hidden_size * 2, seq_len)
        if flag == 1:
            lstm_output = F.max_pool1d(lstm_output, kernel_size=76)
        else:
            lstm_output = F.max_pool1d(lstm_output, kernel_size=78)
        pooled_output = lstm_output.squeeze(2)  # Remove seq_len dimension, output shape: (batch_size, hidden_size * 2)

        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Pass through fully connected layer and sigmoid to get output
        output = self.fc(pooled_output)
        output = self.sigmoid(output)
        
        return output
