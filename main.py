######################################################
# Christoph Aurnhammer, 2019                         #
# Pertaining to Aurnhammer, Frank (2019)             #
# Comparing gated and simple recurrent neural        #
# networks as models of human sentence processing    #
#                                                    #
# Maintained at github.com/caurnhammer/gated_cells   #
# Adpated from pytorch word_language_model           #
######################################################

# coding: utf-8
import argparse
import time
import math
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from random import sample
# import scripts data.py and model.py
import data
import model

def parse_args():
    # parse command line arguments, returned as attribute to object "args"
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
    parser.add_argument('--save', type=str, default='./output/',
                        help='path to save the final model')
    args = parser.parse_args()
    return args

def set_torch_seed(seed):
    # Set the random seed for reproducibility across model types
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)

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

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len])
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def evaluate(rnn_model, data_source, criterion):
    if args.cuda:
        data_source = data_source.cuda()
    # Turn on evaluation mode
    rnn_model.eval()
    # Define end of sentence index
    eos = get_eos()
    # Initiate variables for loss computation
    ntokens = len(corpus.dictionary)
    total_loss = 0
    data_len = 1
    # Loop through test data
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        for j in range(len(data)):
            if (data[j].data == eos)[0] == True:
                # Cut off data at end of sentence
                data = data[:j,:1]
                targets = targets[:j]
                break
        hidden = rnn_model.init_hidden(eval_batch_size)
        output, hidden = rnn_model(data, hidden)
        total_loss += len(data) * criterion(log_softmax(output.view(-1, ntokens), dim=1), targets).data
        data_len += len(data)
    return total_loss.item() / data_len

def shuffle_train_data(train_data):
    # Randomise training data (according to current seed)
    # Set numpy random seed to current seed
    numpy.random.seed(torch.initial_seed())
    # Convert to numpy array for shuffling (changing the np array changes the torch tensor as well)
    train_data = train_data.cpu()
    train_np = train_data.numpy()
    # Shuffle using numpy methods
    N = args.bptt  # Blocks of N rows
    M, n = train_np.shape[0] // N, train_np.shape[1]  # Parameters for reshape (num sentences, num rows)
    numpy.random.shuffle(train_np.reshape(M, -1, n))
    del train_np
    # After shuffling with numpy, set data to GPU
    if args.cuda:
        train_data = train_data.cuda()
    return train_data

def get_eos():
    if args.cuda:
        a = torch.cuda.LongTensor([dictionary.word2idx['<eos>']])
    else:
        a = torch.LongTensor([dictionary.word2idx['<eos>']])
    return a

def train(rnn_model, model_name, seed_index, criterion):
    # Turn on training mode.
    rnn_model.train()
    total_loss = 0
    start_time = time.time()

    # Define optimize, set initial learning rate & constant momentum
    lr = args.lr
    optimizer = optim.SGD(rnn_model.parameters(), lr=lr, momentum=0.9)

    # Define number of sentences at which to take snapshots
    snapshots = [1000 - 1, 3000 - 1, 10000 - 1, 30000 - 1, 100000 - 1, 300000 - 1, 1000000 - 1, 3000000 - 1,
                 (len(train_data) // args.bptt) - 1]

    # Define end of sentence index
    eos = get_eos()

    # Loop through training data
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # chr: cut off data at end of sentence
        for j in range(len(data)):
            if (data[j].data == eos)[0] == True:
                # = end of sentence is reached - remove filler masking elements
                # input data and targets
                data = data[:j,:1]
                targets = targets[:j]
                break

        # Reset hidden states for new sequence
        hidden = rnn_model.init_hidden(args.batch_size)
        rnn_model.zero_grad() # Set gradients to zero for the optimiser
        output, hidden = rnn_model(data, hidden)

        # Optimize network
        loss = criterion(log_softmax(output.view(-1, ntokens), dim=1), targets)
        loss.backward()
        optimizer.step()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), args.clip)
        total_loss += loss.data

        # Print user feedback
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval # current loss
            elapsed = time.time() - start_time           # elapsed time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        # Anneal learning rate in every 3rd of the training data
        if batch in range(0, len(train_data) // args.bptt, len(train_data) // args.bptt // 3) and batch > 0:
            lr /= math.sqrt(4.0)
            optimizer = optim.SGD(rnn_model.parameters(), lr=lr, momentum=0.9)
            print('Updated learning rate to {}'.format(lr))

        if batch in snapshots:
            with open(args.save+model_name+'_'+str(batch+1)+'_'+str(seed_index), 'wb') as f:
                torch.save(rnn_model, f)
                print('> Saved snapshot at {} sentences to disc'.format(batch+1))
    return rnn_model

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Process corpus
    corpus = data.Corpus(args.data, args.bptt)
    dictionary = corpus.dictionary
    ntokens = len(corpus.dictionary)
    train_data = batchify(corpus.train, args.batch_size)
    eval_batch_size = 1
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    # Generate reusable random seeds
    seeds = sample(range(0, 4999), 6)
    seed_indices = range(0, 6)
    print('> chr: The seeds are {}, the seed indices are {}'.format(seeds, [ind for ind in seed_indices]))

    # Loop through seeds (corresponding to models with same sentence order and same initial weights)
    for seed_index, seed in zip(seed_indices, seeds):
        # Set the torch random seed
        torch.manual_seed(seed)

        # Randomise sentence order in train data (using current seed)
        train_data = shuffle_train_data(train_data)

        # Initialise weights for all model types (using current seed)
        arche = model.arche_RNN('LSTM', ntokens, args.emsize, args.nhid, args.nlayers)
        arche.init_weights()

        # For this sentence order and these weights, create on of each RNN types
        models = ['RNN_TANH', 'GRU', 'LSTM']
        for model_name in models:
            # Initialise the rnn model
            rnn_model = model.RNNModel(model_name, ntokens, args.emsize, args.nhid, args.nlayers,
                                       arche.encoder.weight.data, arche.rnn.all_weights, arche.predecoder.bias.data,
                                       arche.predecoder.weight.data, arche.decoder.bias.data, arche.decoder.weight.data)
            if args.cuda:
                rnn_model.cuda()
            print('Initialised Model:', rnn_model.parameters)

            # Define common criterion for training, validating, testing
            criterion = torch.nn.NLLLoss()
            # Loop through epochs.
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                rnn_model = train(rnn_model, model_name, seed_index, criterion)
                # Evaluate on validation data after each epoch
                val_loss = evaluate(rnn_model, val_data, criterion)
                print('=' * 89)
                print('| Validation | loss {:5.2f} | ppl {:8.2f}'.format(
                    val_loss, math.exp(val_loss)))
                print('=' * 89)

            # Evaluate on test data
            test_loss = evaluate(rnn_model, test_data, criterion)
            print('=' * 89)
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss)))
            print('=' * 89)
