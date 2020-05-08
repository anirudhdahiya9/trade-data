import torch
from torch import nn
import torch.nn.functional as F
import logging


logger = logging.getLogger('TEXT CLASSIFIER')


class ConvClassifier(nn.Module):
    def __init__(self, num_classes=6, wvocab_size=50000, wv_dim=100,
                 charvocab_size=64, cv_size=50,
                 hdim=256, embedding_weights=None, char_pad_idx=0):
        super(ConvClassifier, self).__init__()

        if embedding_weights is not None:
            self.wrd_embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_weights), freeze=True)
        else:
            self.wrd_embedding = nn.Embedding(wvocab_size, wv_dim)

        self.char_embedding = nn.Embedding(charvocab_size, cv_size, padding_idx=char_pad_idx)

        num_filters = 10
        self.c2 = nn.Conv1d(cv_size, num_filters, 2)
        self.c3 = nn.Conv1d(cv_size, num_filters, 3)
        self.c4 = nn.Conv1d(cv_size, num_filters, 4)

        # self.mlp = nn.Sequential(nn.Linear(wv_dim + 3 * num_filters, hdim),
        #                          nn.ReLU(),
        #                          nn.Linear(hdim, num_classes))

        self.mlp = nn.Sequential(nn.Linear(3 * num_filters, hdim),
                                 nn.ReLU(),
                                 nn.Linear(hdim, num_classes))

        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, tk_ids, tk_lens, char_ids, char_lens):

        char_embs = self.char_embedding(char_ids).permute(0, 2, 1)
        char_conv2 = self.c2(char_embs)
        p2 = F.max_pool1d(char_conv2, char_conv2.size()[-1]).squeeze(-1)
        char_conv3 = self.c3(char_embs)
        p3 = F.max_pool1d(char_conv3, char_conv3.size()[-1]).squeeze(-1)
        char_conv4 = self.c4(char_embs)
        p4 = F.max_pool1d(char_conv4, char_conv4.size()[-1]).squeeze(-1)
        conv_rep = torch.cat((p2, p3, p4), dim=1)

        '''
        wrd_embs = self.wrd_embedding(tk_ids)
        emb_sum = torch.sum(wrd_embs, 1)
        emb_lens = tk_lens.unsqueeze(1).repeat_interleave(wrd_embs.size()[-1], dim=1)
        emb_avg = emb_sum/emb_lens
        
        hidden_rep = torch.cat((conv_rep, emb_avg), dim=1)
        '''
        logits = self.mlp(conv_rep)

        return logits

    def get_loss(self, tk_ids, tk_lens, char_ids, char_lens, labels):
        logits = self.forward(tk_ids, tk_lens, char_ids, char_lens)
        return self.lossfn(logits, labels), logits
