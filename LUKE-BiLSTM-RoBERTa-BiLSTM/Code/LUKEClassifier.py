import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel, T5Tokenizer, T5EncoderModel, LukeForSequenceClassification
import torch.nn.functional as F

class LUKEClassifier(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.1):
        super(LUKEClassifier, self).__init__()
        
        # Load the luke model
        self.luke = LukeForSequenceClassification.from_pretrained("../luke-base")
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=768,  # Input dimension: luke's hidden layer dimension
                          hidden_size=hidden_size,  # LSTM hidden layer dimension
                          num_layers=3,  # Number of LSTM layers
                          bidirectional=True,  # Bidirectional LSTM
                          batch_first=True)  # Input tensor format is (batch_size, seq_length, feature_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer (Input dimension is LSTM's output dimension, 2 times hidden_size because it's bidirectional)
        self.fc = nn.Linear(hidden_size*2, 1)
        
        # Sigmoid activation function (For binary classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, flag=0):
        # Get the output from luke
        outputs = self.luke(input_ids=input_ids, attention_mask=attention_mask, 
                           output_hidden_states=True, 
                           output_attentions=True)
        
        # luke's last_hidden_state, shape is (batch_size, seq_len, hidden_size)
        outputs = outputs.hidden_states[12]
        # print(outputs.shape)
        
        # Pass luke's output to LSTM layer
        lstm_output, _ = self.lstm(outputs)  # lstm_output shape is (batch_size, seq_len, hidden_size * 2)
        
        # Pooling the LSTM output (Using max pooling here)
        lstm_output = lstm_output.permute(0, 2, 1)  # Transpose, output shape is (batch_size, hidden_size * 2, seq_len)
        if flag == 1:
            lstm_output = F.max_pool1d(lstm_output, kernel_size=76)
        else:
            lstm_output = F.max_pool1d(lstm_output, kernel_size=78)
        pooled_output = lstm_output.squeeze(2)  # Remove the seq_len dimension, output shape is (batch_size, hidden_size * 2)

        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Through fully connected layer and sigmoid to output
        output = self.fc(pooled_output)
        output = self.sigmoid(output)
        
        return output
