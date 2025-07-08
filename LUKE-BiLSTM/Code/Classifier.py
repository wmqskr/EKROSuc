import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel, T5Tokenizer, T5EncoderModel, LukeForSequenceClassification
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.1):
        super(Classifier, self).__init__()
        
        # Load Luke model
        self.luke = LukeForSequenceClassification.from_pretrained("../luke-base")
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=768,  # Input dimension: hidden size of Luke
                          hidden_size=hidden_size,  # Hidden size of LSTM
                          num_layers=3,  # Number of LSTM layers
                          bidirectional=True,  # Bidirectional LSTM
                          batch_first=True)  # Input tensor format: (batch_size, seq_length, feature_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer (input dimension is the output dimension of LSTM, 2 times hidden_size because it is bidirectional)
        self.fc = nn.Linear(hidden_size*2, 1)
        
        # Sigmoid activation function (used for binary classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, flag=0):
        # Get output from Luke
        outputs = self.luke(input_ids=input_ids, attention_mask=attention_mask, 
                           output_hidden_states=True, 
                           output_attentions=True)
        
        # Last hidden state of Luke, shape: (batch_size, seq_len, hidden_size)
        outputs = outputs.hidden_states[12]
        # print(outputs.shape)
        
        # Pass Luke's output to LSTM layer
        lstm_output, _ = self.lstm(outputs)  # LSTM output shape: (batch_size, seq_len, hidden_size * 2)
        
        # Pooling of LSTM output (using max pooling here)
        # pooled_output = lstm_output.mean(dim=1)  # Mean pooling over seq_len dimension, output shape: (batch_size, hidden_size * 2)
        lstm_output = lstm_output.permute(0, 2, 1)  # Transpose, output shape: (batch_size, hidden_size * 2, seq_len)
        if flag == 1:
            lstm_output = F.max_pool1d(lstm_output, kernel_size=76)
        else:
            lstm_output = F.max_pool1d(lstm_output, kernel_size=78)
        pooled_output = lstm_output.squeeze(2)  # Remove seq_len dimension, output shape: (batch_size, hidden_size * 2)

        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Through fully connected layer and sigmoid to get output
        output = self.fc(pooled_output)
        output = self.sigmoid(output)
        
        return output
