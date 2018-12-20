# coding: utf-8
import argparse
import time
import math
import numpy
import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import data

parser = argparse.ArgumentParser(description='PyTorch ENCOW RNN/GRU/LSTM Language Model')
parser.add_argument('--data', type=str, default='./corpus/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.0025,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=41,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./output/',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    # chr: define end of sentence index
    if args.cuda:
        a = torch.cuda.LongTensor([dictionary.word2idx['<eos>']])
    else:
        a = torch.LongTensor([dictionary.word2idx['<eos>']])
    data_len = 1
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        #chr: cut off data at end of sentence
        for j in range(len(data)):
            if (data[j].data == a)[0] == True:
                data = data[:j,:1]
                targets = targets[:j]
                break
        hidden = model.init_hidden(eval_batch_size)
        data_len += len(data)
        output, hidden = model(data, hidden)
        total_loss += len(data) * crit(log_softmax(output.view(-1, ntokens), dim=1), targets).data
    return total_loss[0] / data_len

def train(seed_index):
    # Turn on training mode which enables dropout.
    global model
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_decay = True

    #chr: define end of sentence index
    if args.cuda:
       a = torch.cuda.LongTensor([dictionary.word2idx['<eos>']])
    else:
       a = torch.LongTensor([dictionary.word2idx['<eos>']])
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # chr: cut off data at end of sentence
        for j in range(len(data)):
            if (data[j].data == a)[0] == True: # = end of sentence is reached - remove filler masking elements
                data = data[:j,:1]
                targets = targets[:j]
                break

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = model.init_hidden(args.batch_size)
        model.zero_grad()
        output, hidden = model(data, hidden)

        # training
        loss = crit(log_softmax(output.view(-1, ntokens), dim=1), targets)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        # in every nth of the training data: anneal learning rate
        if batch in range(0, len(train_data) // args.bptt, len(train_data) // args.bptt // 3) and batch > 0:
            args.lr /= math.sqrt(4.0)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            print('Updated learning rate to {}'.format(args.lr))

        snapshots = [1000-1, 3000-1, 10000-1, 30000-1, 100000-1, 300000-1, 1000000-1, 3000000-1, (len(train_data)//args.bptt)-1]
        if batch in snapshots:
            with open(args.save+args.model+'_'+str(batch+1)+'_'+str(seed_index), 'wb') as f:
                torch.save(model, f)
                print('> Saved snapshot at {} sentences to disc'.format(batch+1))

## EXECUTION
# Process corpus
corpus = data.Corpus(args.data, args.bptt)
dictionary = corpus.dictionary
eval_batch_size = 1
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
ntokens = len(corpus.dictionary)

from random import sample
seeds = sample(range(0, 4999), 6)
seed_indices = range(0, 6)
print('> chr: The seeds are {}, the seed indices are {}'.format(seeds, [ind for ind in seed_indices]))
for seed_index, seed in zip(seed_indices, seeds):
    # Set the random seed manually
    args.seed = seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # Randomise training data (according to current seed)
    # Set numpy random seed to current seed
    numpy.random.seed(seed)
    # Convert to numpy array for shuffling (changing the np array changes the torch tensor as well)
    train_data = train_data.cpu()
    train_np = train_data.numpy()
    # Shuffle using numpy methods
    N = args.bptt  # Blocks of N rows
    M, n = train_np.shape[0] // N, train_np.shape[1] # Parameters for reshape (num sentences, num rows)
    numpy.random.shuffle(train_np.reshape(M,-1,n))
    del train_np

    # After shuffling with numpy, set data to GPU
    if args.cuda:
        train_data = train_data.cuda()

    # Initialise weights for all model types (using current seed)
    import model
    archetype = model.arche_RNN('LSTM', ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
    encoder_weights, predecoder_bias, predecoder_weights, decoder_bias, decoder_weight = archetype.init_weights()
    recurrent_weights = archetype.rnn.all_weights
    del archetype

    models = ['RNN_TANH', 'GRU', 'LSTM']
    for model_name in models:
        import model

        args = parser.parse_args()
        args.model = model_name
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
                               encoder_weights, recurrent_weights, predecoder_bias, predecoder_weights,
                               decoder_bias, decoder_weight)

        if args.cuda:
            model.cuda()
        print('Initialised Model:', model.parameters)
        crit = torch.nn.NLLLoss()

        # Loop over epochs.
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            best_val_loss = None
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train(seed_index)
                #val_loss = evaluate(val_data)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Run on test data.
        if args.cuda:
            test_data = test_data.cuda()
        test_loss = evaluate(test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
        del model
