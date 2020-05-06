import torch
from torch import nn
import torch.nn.functional as F


class ConvClassifier:
    def __init__(self, num_classes=6, wvocab_size=50000, wv_dim=100,
                 charvocab_size=64, cv_size=50,
                 hdim=256, embedding_weights=None):

        if embedding_weights:
            self.wrd_embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
        else:
            self.wrd_embedding = nn.Embedding(wvocab_size, wv_dim)

        self.char_embedding = nn.Embedding(charvocab_size, cv_size)

        num_filters = 10
        self.c2 = nn.Conv1d(cv_size, num_filters, 2)
        self.c3 = nn.Conv1d(cv_size, num_filters, 3)
        self.c4 = nn.Conv1d(cv_size, num_filters, 4)

        self.mlp = nn.Sequential(nn.Linear(wv_dim + 3 * num_filters, hdim),
                                 nn.ReLU(),
                                 nn.Linear(hdim, num_classes))

        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, tk_ids, char_ids):

        char_embs = self.char_embedding(char_ids).permute(0, 2, 1)
        char_conv2 = self.c2(char_embs)
        p2 = F.max_pool1d(char_conv2, char_conv2.size()[-1]).squeeze(-1)
        char_conv3 = self.c3(char_embs)
        p3 = F.max_pool1d(char_conv3, char_conv3.size()[-1]).squeeze(-1)
        char_conv4 = self.c4(char_embs)
        p4 = F.max_pool1d(char_conv4, char_conv4.size()[-1]).squeeze(-1)
        conv_rep = torch.cat((p2, p3, p4), dim=1)

        wrd_embs = self.wrd_embedding(tk_ids).permute(0, 2, 1)
        emb_avg = F.avg_pool1d(wrd_embs, wrd_embs.size()[-1]).squeeze(-1)

        hidden_rep = torch.cat((conv_rep, emb_avg), dim=1)

        logits = self.mlp(hidden_rep)

        return logits

    def get_loss(self, tk_ids, char_ids, labels):
        logits = self.forward(tk_ids, char_ids)
        return self.lossfn(logits, labels), logits
