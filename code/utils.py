import unidecode
import os
from torch.utils.data import Dataset
import torch
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_confusion_matrix(ground_truth, predictions, label_list, plot_path):
    # Normalized along rows (true labels), cols are predicted labels
    cmatrix = confusion_matrix(ground_truth, predictions, normalize='true') * 100
    df_cm = pd.DataFrame(cmatrix, label_list, label_list)
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plt.savefig(plot_path, bbox_inches="tight")
    return cmatrix


def clean_string(inp_string, lint_ascii, case_lower):
    inp_string = inp_string.strip()
    if lint_ascii:
        inp_string = unidecode.unidecode(inp_string)
    if case_lower:
        inp_string = inp_string.lower()
    return inp_string


def prepare_emb_matrix(emb_path, tokenizer):
    # Read the glove matrix
    glove_vocab = {}
    with open(emb_path) as f:
        for line in f:
            token = line[:line.find(' ')]
            glove_vocab[token] = np.fromstring(line[line.find(' ') + 1:].strip(), sep=' ')

    # Create the emb matrix
    emb_matrix = np.ndarray(shape=(len(tokenizer.word_vocab), glove_vocab[token].size))
    emb_matrix[tokenizer.word_vocab[tokenizer.pad_token], :] = 0.0

    # Initialize rows with glove vectors
    for token in tokenizer.word_vocab:
        if token not in (tokenizer.unk_token, tokenizer.pad_token):
            emb_matrix[tokenizer.word_vocab[token]] = glove_vocab[token]

    return emb_matrix


class Tokenizer:
    def __init__(self, word_vocab, char_vocab, label2id, unk_token='<unk>', pad_token='<pad>', lint_ascii=True, case_lower=True):

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

        self.unk_token = unk_token
        self.pad_token = pad_token

        # Add padding to word and character vocabs
        self.word_vocab[unk_token] = len(self.word_vocab)
        self.word_vocab[pad_token] = len(self.word_vocab)
        self.char_vocab[unk_token] = len(self.char_vocab)
        self.char_vocab[pad_token] = len(self.char_vocab)

        self.rev_word_vocab = self._reverse_vocab(word_vocab)
        self.rev_char_vocab = self._reverse_vocab(char_vocab)

        self.label2id = label2id
        self.id2label = self._reverse_vocab(label2id)

        self._lint_ascii = lint_ascii
        self._case_lower = case_lower

    def _reverse_vocab(self, vocab):
        return dict((item[1], item[0]) for item in vocab.items())

    @classmethod
    def from_datadir(cls, data_dir, pretrained_vocab, lint_ascii=True, case_lower=True):

        train_files = filter(lambda x: x.startswith('train'), os.listdir(data_dir))
        words_vocab = set()
        categories = []
        for fil in train_files:
            categories.append('_'.join(fil.split('.')[0].split('_')[1:]))
            with open(os.path.join(data_dir, fil)) as f:
                lines = set(f.read().strip().split('\n'))
            clean_lines = [wrd for line in lines for wrd in clean_string(line, lint_ascii, case_lower).split()]
            words_vocab.update(clean_lines)

        words_vocab.intersection_update(pretrained_vocab)

        char_vocab = set()
        for word in words_vocab:
            char_vocab.update(word)
        char_vocab.add(' ')

        words_vocab = dict(zip(words_vocab, range(len(words_vocab))))
        char_vocab = dict(zip(char_vocab, range(len(char_vocab))))

        label2id = dict(zip(categories, range(len(categories))))

        return cls(words_vocab, char_vocab, label2id, unk_token='<unk>', pad_token='<pad>', lint_ascii=lint_ascii,
                   case_lower=case_lower)

    def encode(self, inp_string):

        inp_string = clean_string(inp_string, self._lint_ascii, self._case_lower)
        character_ids = [self.char_vocab[c] if c in self.char_vocab else self.char_vocab[self.unk_token] for c in inp_string]
        tokens = inp_string.split()
        tokens_ids = [self.word_vocab[tk] if tk in self.word_vocab else self.word_vocab[self.unk_token] for tk in tokens]
        return tokens_ids, character_ids

    def decode_tokens(self, token_ids):
        return ' '.join([self.rev_word_vocab[tkid] for tkid in token_ids])

    def decode_chars(self, char_ids):
        return ''.join([self.rev_char_vocab[char_id] for char_id in char_ids])


class TextDataset(Dataset):
    def __init__(self, samples, labels):
        super(Dataset).__init__()
        self.tkids = [torch.LongTensor(sample[0]) for sample in samples]
        self.charids = [torch.LongTensor(sample[1]) for sample in samples]
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, item):
        return self.tkids[item], self.charids[item], self.labels[item]

    def __len__(self):
        return self.labels.size()[0]
