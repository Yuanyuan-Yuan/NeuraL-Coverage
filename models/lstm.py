import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBase(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMBase, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # print('x: ', x.size())
        embedded = self.dropout(self.embedding(x))    
        output, (hidden, cell) = self.lstm(embedded) 
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

def lstm(pretrained):
    if pretrained == True:
        assert False, 'Please specify the path of pretrained weights'
    else:
        # INPUT_DIM = len(TEXT.vocab)
        INPUT_DIM = 25002
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = False
        DROPOUT = 0.5
        model = LSTMBase(
            vocab_size=INPUT_DIM,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT
        )
    return model