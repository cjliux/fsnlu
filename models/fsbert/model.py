#coding: utf-8
"""
    @author: cjliux@gmail.com
"""
import os, sys
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
from .modeling import BertForTaskNLU
# from .tokenization import BertTokenizer
from .optimization import BertAdam, WarmupLinearSchedule
from utils.vocab import Vocab

logger = logging.getLogger()


class Model(nn.Module):

    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args

        self.domain_map = Vocab.from_file(os.path.join(args.data_path, "domains.txt"))
        self.intent_map = Vocab.from_file(os.path.join(args.data_path, "intents.txt"))
        self.slots_map = Vocab.from_file(os.path.join(args.data_path, "slots.txt"))
        self.label_vocab = Vocab.from_file(os.path.join(args.data_path, "label_vocab.txt"))
        self.bin_label_vocab = Vocab.from_file(os.path.join(args.data_path, "bin_label_vocab.txt"))

        self.tokenizer = tokenizer

        label_list = {
            "domain": copy.deepcopy(self.domain_map.index2word),
            "intent": copy.deepcopy(self.intent_map.index2word),
            "slots": copy.deepcopy(self.slots_map.index2word),
            "label_vocab": self.label_vocab,
        }

        self.bert_nlu = BertForTaskNLU.from_pretrained(
            os.path.join(args.bert_dir, "pytorch_model.bin"),
            os.path.join(args.bert_dir, "bert_config.json"),
            label_list=label_list,
            max_seq_len=args.max_seq_length)
        self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=False)

        self.crf_layer = CRF(self.label_vocab)

    def forward(self, batch):
        inputs = batch["model_input"]
        padded_seqs = inputs["padded_seqs"].cuda() 
        seq_lengths = inputs["seq_lengths"].cuda() 

        # seq_lengths
        max_len = seq_lengths.max().item()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        attn_mask = (idxes.cuda() < seq_lengths.unsqueeze(1)).float()

        dom_logits, int_logits, sl_logits = self.bert_nlu(
            input_ids=padded_seqs, 
            #token_type_ids=inputs["segment_ids"],
            attention_mask=attn_mask)
        return { "dom_logits": dom_logits, 
            "int_logits": int_logits, "sl_logits": sl_logits}

    def compute_loss(self, batch, fwd_dict):
        inputs = batch["model_input"]
        dom_idx, int_idx = inputs["dom_idx"].cuda(), inputs["int_idx"].cuda()
        loss = (5 * self.loss_fct(fwd_dict["dom_logits"], dom_idx) 
                + 3 * self.loss_fct(fwd_dict["int_logits"], int_idx))

        padded_y = inputs["padded_y"].cuda()
        loss += 2 * self.crf_layer.compute_loss(fwd_dict["sl_logits"], padded_y)
        return loss.mean()

    def predict(self, batch, fwd_dict):
        dom_pred = torch.argmax(fwd_dict["dom_logits"], dim=-1).detach().cpu().numpy()
        int_pred = torch.argmax(fwd_dict["int_logits"], dim=-1).detach().cpu().numpy()
        crf_pred = self.crf_layer(fwd_dict["sl_logits"])
        lbl_pred = [crf_pred[i, :l].data.cpu().numpy() 
                            for i, l in enumerate(batch["model_input"]["seq_lengths"])]
        return dom_pred, int_pred, lbl_pred


class CRF(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via
    backpropagation. 
    """
    def __init__(self, label_vocab):
        super().__init__()
        self.label_vocab = label_vocab
        self.num_tags = label_vocab.n_words
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats)

    def compute_loss(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and 
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags (padded sequences)
        Returns:
            Negative log likelihood [batch size] 
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError('tags must be 2-d but got {}-d'.format(tags.shape))

        if feats.shape[:2] != tags.shape:
            raise ValueError('First two dimensions of feats and tags must match ', feats.shape, tags.shape)

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        # -ve of l()
        return - log_probability

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        batch_size = feats.shape[0]

        # Compute feature scores
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # print(feat_score.size())

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for 
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))

        a = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1) # [batch_size, 1, num_tags]
            a = torch.logsumexp(a.unsqueeze(-1) + transitions + feat, 1) # [batch_size, num_tags]

        return torch.logsumexp(a + self.stop_transitions.unsqueeze(0), 1) # [batch_size]

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(self.num_tags, num_tags))
        
        v = feats[:, 0] + self.start_transitions.unsqueeze(0) # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(0) # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i] # [batch_size, num_tags]
            v, idx = (v.unsqueeze(-1) + transitions).max(1) # [batch_size, num_tags], [batch_size, num_tags]
            
            paths.append(idx)
            v = (v + feat) # [batch_size, num_tags]

        
        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)
