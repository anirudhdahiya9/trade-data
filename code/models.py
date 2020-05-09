import torch
from torch import nn
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)


class ConvClassifier(nn.Module):
    def __init__(self, num_classes=6, wvocab_size=50000, wv_dim=100,
                 charvocab_size=64, cv_size=50,
                 hdim=256, embedding_weights=None, char_pad_idx=0, model_type='hybrid'):
        super(ConvClassifier, self).__init__()

        if embedding_weights is not None:
            self.wrd_embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_weights), freeze=True)
        else:
            self.wrd_embedding = nn.Embedding(wvocab_size, wv_dim)

        self.char_embedding = nn.Embedding(charvocab_size, cv_size, padding_idx=char_pad_idx)

        num_filters = 10
        self.c2 = nn.Sequential(nn.Conv1d(cv_size, num_filters, 2), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv1d(cv_size, num_filters, 3), nn.ReLU())
        self.c4 = nn.Sequential(nn.Conv1d(cv_size, num_filters, 4), nn.ReLU())


        if model_type=='conv':
            hrep_dim = 3*num_filters
        elif model_type=='bow':
            hrep_dim = wv_dim
        else:
            hrep_dim = wv_dim + 3*num_filters

        self.mlp = nn.Sequential(nn.Linear(hrep_dim, hdim),
                                 nn.ReLU(),
                                 nn.Linear(hdim, num_classes))

        self.lossfn = nn.CrossEntropyLoss()
        self._model_type = model_type

        self.hyperparams = {
            'model_type': self._model_type,
            'num_classes': num_classes,
            'wvocab_size':self.wrd_embedding.num_embeddings,
            'wv_dim': self.wrd_embedding.embedding_dim,
            'charvocab_size': self.char_embedding.num_embeddings,
            'cv_size': self.char_embedding.embedding_dim,
            'hdim': hdim,
            'embedding_weights': None,
            'char_pad_idx': char_pad_idx
        }

    def _charconv(self, char_ids):
        char_embs = self.char_embedding(char_ids).permute(0, 2, 1)
        char_conv2 = self.c2(char_embs)
        p2 = F.max_pool1d(char_conv2, char_conv2.size()[-1]).squeeze(-1)
        char_conv3 = self.c3(char_embs)
        p3 = F.max_pool1d(char_conv3, char_conv3.size()[-1]).squeeze(-1)
        char_conv4 = self.c4(char_embs)
        p4 = F.max_pool1d(char_conv4, char_conv4.size()[-1]).squeeze(-1)
        conv_rep = torch.cat((p2, p3, p4), dim=1)
        return conv_rep

    def _wrd_emb_avg(self, tk_ids, tk_lens):
        wrd_embs = self.wrd_embedding(tk_ids)
        emb_sum = torch.sum(wrd_embs, 1)
        emb_lens = tk_lens.unsqueeze(1).repeat_interleave(wrd_embs.size()[-1], dim=1)
        emb_avg = emb_sum/emb_lens
        return emb_avg

    def forward(self, tk_ids, tk_lens, char_ids, char_lens):

        if self._model_type == 'hybrid':
            conv_rep = self._charconv(char_ids)
            emb_avg = self._wrd_emb_avg(tk_ids, tk_lens)
            hidden_rep = torch.cat((conv_rep, emb_avg), dim=1)
        elif self._model_type == 'bow':
            hidden_rep = self._wrd_emb_avg(tk_ids, tk_lens)
        elif self._model_type == 'conv':
            hidden_rep = self._charconv(char_ids)

        logits = self.mlp(hidden_rep)

        return logits

    def get_loss(self, tk_ids, tk_lens, char_ids, char_lens, labels):
        logits = self.forward(tk_ids, tk_lens, char_ids, char_lens)
        return self.lossfn(logits, labels), logits

    def save_model(self, save_path):
        model_save_dict = self.state_dict()
        model_save_dict['hyperparams'] = self.hyperparams
        torch.save(model_save_dict, save_path)

    @classmethod
    def load_model(cls, save_path):
        state_dict = torch.load(save_path)
        model = cls(**state_dict['hyperparams'])
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys!=[]:
            logger.error(f"ConvModel received state_dict with missing keys: {missing_keys}, returning None.")
            return None
        return model
