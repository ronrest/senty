import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    """ An LSTM Model for classification task"""
    def __init__(self, n_vocab, embed_size, h_size, n_layers=1, dropout=0.5):
        super(Model, self).__init__()
        
        self.n_vocab = n_vocab
        self.embed_size = embed_size
        self.h_size = h_size
        self.n_layers = n_layers
        self.alpha = 0.001  # learning rate
        self.batch_size = 1
        self.classes = 2
        
        self.embeddings = nn.Embedding(n_vocab, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=h_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout
                            )
        
        self.classifier = nn.Linear(h_size, 2)
        
        # SPECIFY LOSS AND OPTIMIZER FUNCTIONS
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha)
    
    def forward(self, input, hidden):
        timesteps = input.size()[0]
        
        input = self.embeddings(input)
        # Reshape input dimensions to (batch_size, timesteps, embed_size)
        input = input.view(self.batch_size, timesteps, self.embed_size)
        
        lstm_out, hidden = self.lstm(input, hidden)

        # Use the very final output for classification
        lstm_out  = lstm_out[:, -1, :].view(self.batch_size, self.h_size)
        output = self.classifier(lstm_out)
        
        return output, hidden
    
    def init_hidden(self):
        self.batch_size = 1
        # Note batch is always second index for hidden EVEN when batch_first
        # argument was set for the LSTM.
        return (Variable(torch.zeros(self.n_layers, self.batch_size, self.h_size)),
                Variable(torch.zeros(self.n_layers, self.batch_size, self.h_size)))
    
    def update_learning_rate(self, alpha):
        """ Updates the learning rate without resetting momentum."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = alpha
        self.alpha = alpha

