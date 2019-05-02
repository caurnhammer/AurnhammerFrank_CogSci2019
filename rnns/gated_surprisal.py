###############################################################
# This code computes surprisal for the RNN language models    #
# described in Aurnhammer, Frank (2019, CogSci proceedings)   #
# (Christoph Aurnhammer, 05.04.2019                           #
###############################################################

import argparse
import os
import pandas
import glob
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from math import exp, isnan
# Requires script data.py
import data


def get_checkpoints(check_paths, model_types):
    ch_points = []
    for repetition in range(0, 6):
        for modelname in model_types:
            found_checkpoints = glob.glob(check_paths+modelname+'*'+str(repetition))
            if found_checkpoints != []:
                for cp in sorted(found_checkpoints, key = numericalSort):
                    ch_points.append(cp)
    return ch_points


def numericalSort(value):
    # Numerical sort from here on
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ENCOW Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='corpus/',
                        help='location of the data corpus')
    # chr: this needs to match the longest sequence in the training data!
    parser.add_argument('--bptt', type=int, default=41,
                        help='sequence length')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    args = parser.parse_args()
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return args


def prepare_outfile(out):
    # Read test sentences
    with open('corpus/test.txt') as inputfile:
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
    return df


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
    data = Variable(source[i:i + seq_len])
    target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
    return data, target


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


def add_surprisal(total_surprisal):
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


def evaluate(surprisal_values):
    N = 0
    Psum = 0

    for surp in surprisal_values:
        if isnan(surp[1]):
            pass
        else:
            N += 1
            Psum += -surp[1]
    print("Evaluated: Perplexity {}".format(round(exp(-1 / N * Psum), 4)))
    return round(exp(-1 / N * Psum), 4)


def store_eval(ch_points, ppl):
    models = []
    points = []
    rep = []
    for i, c in enumerate(ch_points):
        ch_points[i] = ch_points[i].replace("RNN_TANH", "SRN")
        ch_points[i] = ch_points[i].strip('./output/')
        ch_points[i] = ch_points[i].split('_')
        models.append(ch_points[i][0])
        points.append(ch_points[i][1])
        rep.append(ch_points[i][2])

    dt = pandas.DataFrame()
    dt['model'] = models
    dt['snapshot'] = points
    dt['repetition'] = rep
    dt['perplexity'] = ppl
    dt.to_csv('test_perplexity.txt', sep='\t', index=False)
    print("Stored perplexity values to file test_perplexity.txt")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Get checkpoint absolute paths from dir
    checkpoints_path = './output/'
    models = ['RNN_TANH', 'GRU', 'LSTM']
    checkpoints = get_checkpoints(check_paths=checkpoints_path, model_types=models)

    # Prepare output file
    outfile = './gated_surprisal.txt'
    dataframe = prepare_outfile(outfile)

    # Load corpus
    print('Loading corpus from {}'.format(args.data))
    corpus = data.Corpus(args.data, args.bptt)
    dictionary = corpus.dictionary
    ntypes = len(corpus.dictionary)
    criterion = nn.CrossEntropyLoss()

    # Collect perplexity values
    test_ppl = []

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

        ## chr: Testing
        eval_batch_size = 1
        test_data = batchify(corpus.test, eval_batch_size)

        # Get surprisal
        surprisal = get_surprisal(test_data)
        add_surprisal(surprisal)

        # Get test loss / perplexity
        test_ppl.append(evaluate([y for x in surprisal for y in x]))

    # Store perplexity
    store_eval(checkpoints, test_ppl)

    # Exclude columns
    df = pandas.read_csv(outfile, delimiter='\t', header=0)
    del df['word']
    df.to_csv(outfile, sep='\t', index=False, float_format='%11.4f')
