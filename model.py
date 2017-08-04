import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    """ An LSTM Model for classification task"""
    def __init__(self, n_vocab, embed_size, h_size, n_layers=1, dropout=0.5, l2=0, average_out=False):
        super(Model, self).__init__()
        
        self.n_vocab = n_vocab
        self.embed_size = embed_size
        self.h_size = h_size
        self.n_layers = n_layers
        self.alpha = 0.001  # learning rate
        self.batch_size = 1
        self.classes = 2
        self.average_out = average_out  # use the average of the steps as input
                                        # to the classifier.
        
        self.embeddings = nn.Embedding(n_vocab, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=h_size,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout
                            )
        
        # FC classification layer with glorot initialization and bias of 0.01
        self.classifier = nn.Linear(h_size, 2)
        init.xavier_normal(self.classifier.weight, gain=1)
        init.constant(self.classifier.bias, 0.01)
        
        # SPECIFY LOSS AND OPTIMIZER FUNCTIONS
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha,
                                          weight_decay=l2)
    
    def forward(self, input, hidden):
        timesteps = input.size()[1]
        
        input = self.embeddings(input)
        # Reshape input dimensions to (batch_size, timesteps, embed_size)
        input = input.view(self.batch_size, timesteps, self.embed_size)
        
        lstm_out, hidden = self.lstm(input, hidden)

        # Use the very final output for classification
        if self.average_out:
            lstm_out = lstm_out.mean(dim=1).view(self.batch_size, self.h_size)
        else:
            lstm_out  = lstm_out[:, -1, :].view(self.batch_size, self.h_size)
        
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        # Note batch is always second index for hidden EVEN when batch_first
        # argument was set for the LSTM.
        return (Variable(torch.zeros(self.n_layers, self.batch_size, self.h_size)),
                Variable(torch.zeros(self.n_layers, self.batch_size, self.h_size)))
    
    def update_learning_rate(self, alpha):
        """ Updates the learning rate without resetting momentum."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = alpha
        self.alpha = alpha
    
    def set_embedding_weights(self, weights):
        """ Given a numpy array of weights, it sets the embedding values """
        self.embeddings.weight.data = torch.Tensor(weights)


# # CODE TO RUN A FORWARD PASS
# model = Model(n_vocab=10, embed_size=5, h_size=7, n_layers=1, dropout=0.5)
# hidden = model.init_hidden( )
# model.zero_grad()
# input = Variable(torch.LongTensor([3,1,7, 4]))
# out, _ = model(input, hidden)

