import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, bptt):
        self.bptt = bptt
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for index, line in enumerate(f):
                words = ['<sos>'] + line.split() + ['<eos>']
                tokens += len(words)
                # Add filler tokens to mask ends-of-sentences
                if len(words) < self.bptt:
                    for i in range(self.bptt-len(words)):
                        tokens += 1
                for word in words:
                    self.dictionary.add_word(word.lower())
                if index in range (0, index+10000, 10000):
                    print('> {} : Building vocabulary. Processed {}, vocabulary size = {}'.format(
                        os.path.basename(path), index,len(self.dictionary)), end='\r')
                last_index = index
        print('> {} : Building vocabulary. Processed {}, vocabulary size = {}'.format(
                os.path.basename(path), last_index+1, len(self.dictionary)))

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for index, line in enumerate(f):
                words = ['<sos>'] + line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word.lower()]
                    token += 1
                # Fill up sentence ends with end-of-sentence indices until max sentence number (indicated by bptt value) is reached.
                # This is to ensure indipendence between sentences
                if len(words) < self.bptt:
                    for i in range(self.bptt-len(words)):
                        ids[token] = self.dictionary.word2idx['<eos>']
                        token += 1
                if index in range(0, index+10000, 10000):
                    print('> {} : Converting data. Processed {}'.format(os.path.basename(path), index), end='\r')
                last_index = index
        print('> {} : Converting data. Processed {}'.format(os.path.basename(path), last_index+1))
        return ids
