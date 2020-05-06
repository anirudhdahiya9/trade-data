from argparse import ArgumentParser
import logging
import os
from utils import Tokenizer, TextDataset, prepare_emb_matrix
from models import ConvClassifier
import pydevd_pycharm


def setup_logging(log_path):
    logger = logging.getLogger('TEXT CLASSIFIER')
    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fhandler = logging.FileHandler(log_path)
    fhandler.setLevel(logging.INFO)
    fhandler.setFormatter(format)

    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(format)

    logger.addHandler(fhandler)
    logger.addHandler(chandler)


def train(model, data, args):
    pass


def preprocess_data(data_dir, split, tokenizer):
    files = filter(lambda x: x.startswith(split), os.listdir(data_dir))
    samples = []
    labels = []
    for fil in files:
        category = '_'.join(fil.split('.')[0].split('_')[1:])
        label = tokenizer.label2id[category]
        with open(os.path.join(data_dir, fil)) as f:
            for line in f:
                samples.append(tokenizer.encode(line))
                labels.append(label)

    dataset = TextDataset(samples, labels)
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser(prog="Text Classifier", description="Training Code for the trade text classifier.")

    parser.add_argument('--remote_debug', type=bool, default=True, help="Flag for the remote debug process.")

    parser.add_argument('--mode', choices=('train', 'infer'), required=True, help="Run mode.")
    parser.add_argument('--lint_ascii', default=True, type=bool, help="Run mode.")
    parser.add_argument('--case_lower', default=True, type=bool, help="Run mode.")

    parser.add_argument('--data_dir', default='../data/splits', type=str, help="Path to the data splits directory.")
    parser.add_argument('--pretrained_vocab_path', default='../data/glove_vocab.txt', type=str, help="Path to file "
                                                                                                     "containing pretrained "
                                                                                                     "vocab.")
    parser.add_argument('--pretrained_glove_path', default='/DATA1/USERS/anirudh/glove6B/glove.6B.100d.txt',
                        type=str, help="Path to the glove embeddings file.")
    parser.add_argument('--log_path', default='../logs/main.log', type=str, help="Path to the log file.")

    args = parser.parse_args()

    if args.remote_debug:
        pydevd_pycharm.settrace('10.1.65.133', port=2134, stdoutToServer=True, stderrToServer=True)

    setup_logging(args.log_path)

    if args.mode == 'train':
        with open(args.pretrained_vocab_path) as f:
            pretrained_vocab = set(f.read().strip().split('\n'))

        tokenizer = Tokenizer.from_datadir(args.data_dir, pretrained_vocab, args.lint_ascii, args.case_lower)

        preprocessed_data = preprocess_data(args.data_dir, 'train', tokenizer)

        emb_matrix = prepare_emb_matrix(args.pretrained_glove_path, tokenizer)
        model = ConvClassifier(wvocab_size=len(tokenizer.word_vocab), charvocab_size=len(tokenizer.char_vocab),
                               embedding_weights=emb_matrix)
        

    pass
