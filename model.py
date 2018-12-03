import torch
import torch.nn as nn
from torch.autograd import Variable

class arche_RNN(nn.Module):
    """A dummy class to return weights and biases. Allows to feed the same weights into different rnn classes."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0, tie_weights=False):
        self.predecoder_out = 400

        super(arche_RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                          options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.predecoder = nn.Linear(nhid, self.predecoder_out)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(self.predecoder_out, ntoken)

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.predecoder.bias.data.fill_(0)
        self.predecoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        return self.encoder.weight.data, self.predecoder.bias.data, self.predecoder.weight.data, \
               self.decoder.bias.data, self.decoder.weight.data

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a tanh pre-decoder, a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout,
                 encoder_weights, recurrent_weights, predecoder_bias, predecoder_weights, decoder_bias, decoder_weight):
        self.predecoder_out = 400

        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            if rnn_type == 'LSTM':
                for index, param in enumerate(self.rnn.parameters()):
                    param.data = torch.FloatTensor(recurrent_weights[0][index].data)
            if rnn_type == 'GRU':
                for index, param in enumerate(self.rnn.parameters()):
                    param.data = torch.FloatTensor(recurrent_weights[0][index][:nhid*3].data)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                      options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            # Reset weights to those fed into the class call
            for index, param in enumerate(self.rnn.parameters()):
                param.data = torch.FloatTensor(recurrent_weights[0][index][:nhid].data)

        self.predecoder = nn.Linear(nhid, self.predecoder_out)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(self.predecoder_out, ntoken)

        # Initialise weights received from class call input
        self.encoder.weight.data = encoder_weights
        self.predecoder.bias.data = predecoder_bias
        self.predecoder.weight.data = predecoder_weights
        self.decoder.bias.data = decoder_bias
        self.decoder.weight.data = decoder_weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output_rnn, hidden = self.rnn(emb, hidden)
        output_predecoder = self.tanh(self.predecoder(output_rnn))
        decoded = self.decoder(output_predecoder.view(output_predecoder.size(0)*output_predecoder.size(1), output_predecoder.size(2)))
        return decoded.view(output_predecoder.size(0), output_predecoder.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

