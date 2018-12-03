import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import data
import os
import pandas
from torch.nn.functional import log_softmax
import math

# Get checkpoint absolute paths from dir
import glob
import re
checkpoints_path = './output/'
models = ['RNN_TANH', 'GRU', 'LSTM']
checkpoints = []
for repetition in range(0, 6):
    for modelname in models:
        found_checkpoints = glob.glob(checkpoints_path+modelname+'*'+str(repetition))
        def numericalSort(value):
            # Numerical sort from here on
            numbers = re.compile(r'(\d+)')
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
        if found_checkpoints != []:
            for cp in sorted(found_checkpoints, key = numericalSort):
                checkpoints.append(cp)

outfile = './gated_surprisal.txt'

parser = argparse.ArgumentParser(description='PyTorch ENCOW Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./smalldata/',
                    help='location of the data corpus')
## chr: this needs to match the longest sequence in the training data!
parser.add_argument('--bptt', type=int, default=41,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# Prepare output file
with open('./smalldata/test.txt') as inputfile:
    inputfile = inputfile.read()
    inputfile = inputfile.replace(' ,', ',')
    inputfile = inputfile.replace(' n\'t', 'n\'t')
    inputfile = inputfile.replace(' \'', '\'')
    inputfile = inputfile.split('\n')
    del inputfile[-1]
    inputfile = [sentence.split(' ') for sentence in inputfile]
for sent_ind, sentence in enumerate(inputfile):
    for word_ind, word in enumerate(sentence):
        if word == ',':
            del inputfile[sent_ind][word_ind]
sent_nr = []
word_pos = []
words = []
for sent_ind, sentence in enumerate(inputfile):
    for word_ind, word in enumerate(sentence):
        sent_nr.append(sent_ind + 1)
        word_pos.append(word_ind + 1)
        words.append(word)

df = pandas.DataFrame()
df['sent_nr'] = sent_nr
df['word_pos'] = word_pos
df['word'] = words
df.to_csv(outfile, sep='\t', index=False)

print('Loading corpus from {}'.format(args.data))
corpus = data.Corpus(args.data, args.bptt)
dictionary = corpus.dictionary
ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()

for index, checkpoint in enumerate(checkpoints):
    print('Checkpoint: ', checkpoint)
    with open(checkpoint, 'rb') as f:
        model = torch.load(f)
    model.eval()

    if args.cuda:
        model.cuda()
    else:
        model.cpu()
    hidden = model.init_hidden(1)

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
        data = Variable(source[i:i + seq_len], volatile=evaluation)
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
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
            # chr: cut off data at end of sentence
            for j in range(len(data)):
                if (data[j].data == a)[0] == True:
                    data = data[:j, :1]
                    targets = targets[:j]
                    break
            data_len += len(data)
            hidden = model.init_hidden(eval_batch_size)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)  # chr: ntokens == vocabulary size
            total_loss += len(data) * criterion(output_flat, targets).data
        return total_loss[0] / data_len

    def get_surprisal(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        ntokens = len(corpus.dictionary)
        if args.cuda:
            a = torch.cuda.LongTensor([dictionary.word2idx['<eos>']])
        else:
            a = torch.LongTensor([dictionary.word2idx['<eos>']])
        # chr: Define list for surprisal values per sentence
        total_surprisal = []
        for x in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, x, evaluation=True)
            # chr: cut off data at end of sentence
            for j in range(len(data)):
                if (data[j].data == a)[0] == True:
                    data = data[:j, :1]
                    targets = targets[:j]
                    break

            hidden = model.init_hidden(eval_batch_size)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            output_sftmx = log_softmax(output_flat, dim=1)

            # Word by word add surprisal values
            sent_surprisal = []
            for target, prob in zip(targets.data, output_sftmx.data):
                sent_surprisal.append([corpus.dictionary.idx2word[target], round(float(prob[target]), 4)*-1])
            del(sent_surprisal[-1])
            total_surprisal.append(sent_surprisal)
        return total_surprisal


    def add_surprisal(total_surprisal):
        def mk_surprisal_list(dataframe, total_surprisal):
            surprisal_col = []
            for row in dataframe.iterrows():
                word_value = total_surprisal[row[1][0]-1][row[1][1]-1]
                word = row[1][2].lower()
                word = word.strip(',')
                if word_value[0] == word:
                    surprisal_col.append(round(word_value[1], 4))
                else:
                    surprisal_col.append(None)
            assert len(dataframe) == len(surprisal_col)
            return surprisal_col

        print('> Adding surprisal to experimental data')
        # Clean up sentences: remove commas and words with clitics (ensures equal lengths of sentences)
        for i, sentence in enumerate(total_surprisal):
            for j, word in enumerate(sentence):
                if word[0] == ',':
                    del (total_surprisal[i][j])
                elif '\'' in word[0]:
                    del (total_surprisal[i][j])

        df = pandas.read_csv(outfile, delimiter='\t', header=0)

        # Add surprisal to dataframe
        df[os.path.basename(checkpoint)+'_surprisal'] = mk_surprisal_list(df, total_surprisal)

        # Clean up extra columns
        try:
            del df['Unnamed: 0']
            del df['Unnamed: 0.1']
        except KeyError:
            pass
        df.to_csv(outfile, sep='\t', index=False)
        print('> Saved updated experimental data to file')


    ## chr: Testing
    eval_batch_size = 1
    test_data = batchify(corpus.test, eval_batch_size)
    valid_data = batchify(corpus.valid, eval_batch_size)


    # Get surprisal
    surprisal = get_surprisal(test_data)
    add_surprisal(surprisal)

    # Get validation loss / perplexity
    valid_loss = evaluate(valid_data)
    print('=' * 89)
    print(
       '| End of training | validation loss {:5.2f} | validation ppl {:8.2f}'.format(valid_loss, math.exp(valid_loss)))
    print('=' * 89)

    # Get test loss / perplexity
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
          test_loss, math.exp(test_loss)))
    print('=' * 89)

# Exclude columns
df = pandas.read_csv(outfile, delimiter='\t', header=0)
del df['word']
df.to_csv(outfile, sep='\t', index=False, float_format='%11.4f')
