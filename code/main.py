from argparse import ArgumentParser
import logging
import os
from utils import Tokenizer, TextDataset, prepare_emb_matrix, plot_confusion_matrix
from models import ConvClassifier
import pydevd_pycharm
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm
import _pickle as pickle

TK_PAD_IDX = None
CHAR_PAD_IDX = None


def setup_logging(log_path):
    logger = logging.getLogger(__name__)
    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fhandler = logging.FileHandler(log_path)
    fhandler.setLevel(logging.INFO)
    fhandler.setFormatter(format)

    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(format)

    logger.addHandler(fhandler)
    logger.addHandler(chandler)
    logger.setLevel(logging.INFO)


def pad_batch(batch_samples):
    tk_ids, char_ids, labels = zip(*batch_samples)
    tk_lens = torch.LongTensor([len(tk) for tk in tk_ids])
    char_lens = torch.LongTensor([len(c) for c in char_ids])
    padded_tkids = pad_sequence(tk_ids, batch_first=True, padding_value=TK_PAD_IDX)
    padded_chars = pad_sequence(char_ids, batch_first=True, padding_value=CHAR_PAD_IDX)
    labels = torch.LongTensor(labels)
    return padded_tkids, tk_lens, padded_chars, char_lens, labels


def evaluate(model, dataset):

    predictions = []
    ground_truth = []
    running_loss = 0.0

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=pad_batch)
    model.eval()
    for batch in dataloader:

        batch = (t.to(args.device) for t in batch)
        tk_ids, tk_lens, char_ids, char_lens, labels = batch
        loss, logits = model.get_loss(tk_ids, tk_lens, char_ids, char_lens, labels)

        predictions.extend(logits.argmax(dim=1).tolist())
        ground_truth.extend(labels.tolist())
        running_loss += loss.item()

    running_loss /= len(dataloader)

    return predictions, ground_truth, running_loss


def train(model, train_dataset, dev_dataset, label_list, args):

    save_path = os.path.join(args.save_path, args.experiment_name+'.ckpt')

    logger.info(f"Entered training")

    tb_writer = SummaryWriter(args.tbwriter_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch, shuffle=True)

    predictions, ground_truth, eval_loss = evaluate(model, dev_dataset)
    logger.info(f"Validation Loss Epoch -1 {eval_loss}")
    logger.info(f"Epoch -1 Evaluation Results\n{classification_report(ground_truth, predictions, target_names=label_list)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_fscore = 0.0
    stopping_epochs = 0

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            batch = (t.to(args.device) for t in batch)
            tk_ids, tk_lens, char_ids, char_lens, labels = batch
            loss, _ = model.get_loss(tk_ids, tk_lens, char_ids, char_lens, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        predictions, ground_truth, eval_loss = evaluate(model, dev_dataset)
        precision, recall, fscore, _ = precision_recall_fscore_support(ground_truth, predictions, average='macro')

        tb_writer.add_scalars('Train Val Loss FScore', {'Train Loss': train_loss,
                                                        'Val Loss': eval_loss,
                                                        'F-Score': fscore}, epoch)
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss {train_loss}")
        logger.info(f"Validation Loss {eval_loss}")
        logger.info(f"Precision: {precision}  Recall: {recall}  FScore: {fscore}")

        if fscore > best_fscore:
            stopping_epochs = 0
            best_fscore = fscore
            model.save_model(save_path)
            logger.info(f'Checkpoint saved at {save_path}')
        else:
            stopping_epochs += 1
            if stopping_epochs > args.early_stopping_threshold:
                logger.info(f'Early Stopping criteria reached at epoch {epoch}')
                break

    model = ConvClassifier.load_model(save_path)
    return model


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

    parser.add_argument('--mode', choices=('train', 'test'), required=True, help="Run mode.")
    parser.add_argument('--model_type', choices=('bow', 'conv', 'hybrid'), required=True, help="Model Type wanted.")
    parser.add_argument('--experiment_name', default='base_training', help="Label for the run.")
    parser.add_argument('--lint_ascii', default=True, type=bool, help="Reduce non-ascii to ascii characters")
    parser.add_argument('--case_lower', default=True, type=bool, help="Lower case all texts")

    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help="Device")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size for training.")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Adam optimizer learning rate")
    parser.add_argument('--num_epochs', default=30, type=int, help="Number of training epochs.")
    parser.add_argument('--early_stopping_threshold', default=5, type=int)

    parser.add_argument('--data_dir', default='../data/splits', type=str, help="Path to the data splits directory.")
    parser.add_argument('--pretrained_vocab_path', default='../data/glove_vocab.txt', type=str, help="Path to file "
                                                                                                     "containing pretrained "
                                                                                                     "vocab.")
    parser.add_argument('--pretrained_glove_path', default='/DATA1/USERS/anirudh/glove6B/glove.6B.100d.txt',
                        type=str, help="Path to the glove embeddings file.")
    parser.add_argument('--tbwriter_path', default='../tblogs', type=str, help="Path to the tensorboard log path.")
    parser.add_argument('--log_path', default='../logs', type=str, help="Path to the log file.")
    parser.add_argument('--plot_path', default='../cmatrices', type=str, help="Path to plot conf matrices")
    parser.add_argument('--save_path', default='../checkpoints/', type=str, help="Path to save model checkpoints.")

    args = parser.parse_args()

    if args.remote_debug:
        pydevd_pycharm.settrace('10.1.65.133', port=2134, stdoutToServer=True, stderrToServer=True)

    # Setup Logging
    log_path = os.path.join(args.log_path, args.experiment_name)
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    argstring = '\n'.join([arg + ':' + str(args.__getattribute__(arg)) for arg in args.__dict__])
    logger.info(f"Running with the following arguments\n{argstring}")

    # Setup checkpoints path
    args.save_path = os.path.join(args.save_path, args.experiment_name)
    os.makedirs(args.save_path, exist_ok=True)

    if args.device=='cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.mode == 'train':
        with open(args.pretrained_vocab_path) as f:
            pretrained_vocab = set(f.read().strip().split('\n'))

        args.tbwriter_path = os.path.join(args.tbwriter_path, args.experiment_name)

        # Tokenizer
        tokenizer = Tokenizer.from_datadir(args.data_dir, pretrained_vocab, args.lint_ascii, args.case_lower)
        label_list = [tokenizer.id2label[i] for i in range(len(tokenizer.id2label))]
        with open(os.path.join(args.save_path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)

        # Order of setting these important, should be set before model and dataset creation
        TK_PAD_IDX = tokenizer.word_vocab[tokenizer.pad_token]
        CHAR_PAD_IDX = tokenizer.char_vocab[tokenizer.pad_token]

        # Setup datasets
        train_dataset = preprocess_data(args.data_dir, 'train', tokenizer)
        dev_dataset = preprocess_data(args.data_dir, 'dev', tokenizer)
        test_dataset = preprocess_data(args.data_dir, 'test', tokenizer)

        # Model Creation
        emb_matrix = prepare_emb_matrix(args.pretrained_glove_path, tokenizer)
        model = ConvClassifier(wvocab_size=len(tokenizer.word_vocab), charvocab_size=len(tokenizer.char_vocab),
                               embedding_weights=emb_matrix, char_pad_idx=CHAR_PAD_IDX, model_type=args.model_type)
        model.to(args.device)

        # Training
        model = train(model, train_dataset, dev_dataset, label_list, args)
        model.to(args.device)

        predictions, ground_truth, running_loss = evaluate(model, test_dataset)
        logger.info(f'Test Results on best model\n'
                     f'{classification_report(ground_truth, predictions, target_names=label_list)}')
        plot_path = os.path.join(args.plot_path, args.experiment_name + '.png')
        cmatrix = plot_confusion_matrix(ground_truth, predictions, label_list, plot_path)
        logging.info(f'Confusion Matrix:\n{cmatrix}')
        logging.info(f'Confusion Matrix saved at {plot_path}')

    elif args.mode == 'test':

        logger.info(f"Entered inference for experiment {args.experiment_name}")

        with open(os.path.join(args.save_path, 'tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)
        label_list = [tokenizer.id2label[i] for i in range(len(tokenizer.id2label))]

        save_path = os.path.join(args.save_path, args.experiment_name + '.ckpt')
        model = ConvClassifier.load_model(save_path).to(args.device)

        # Order of setting these important, should be set before model and dataset creation
        TK_PAD_IDX = tokenizer.word_vocab[tokenizer.pad_token]
        CHAR_PAD_IDX = tokenizer.char_vocab[tokenizer.pad_token]

        test_dataset = preprocess_data(args.data_dir, 'test', tokenizer)
        predictions, ground_truth, running_loss = evaluate(model, test_dataset)
        logger.info(f'Test Results on best model\n'
                    f'{classification_report(ground_truth, predictions, target_names=label_list)}')
        plot_path = os.path.join(args.plot_path, args.experiment_name+'.png')
        cmatrix = plot_confusion_matrix(ground_truth, predictions, label_list, plot_path)
        logger.info(f'Confusion Matrix:\n{cmatrix}')
        logger.info(f'Confusion Matrix saved at {plot_path}')
        pass
