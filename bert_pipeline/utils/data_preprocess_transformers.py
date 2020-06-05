import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from pytorch_transformers import *

class DataPreprocessor(object):
    def __init__(self, model):
        MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
                (GPT2Model,       GPT2Tokenizer,      'gpt2'),
                (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased')]

        if model == -1 or model > 2:
            model_class, tokenizer_class, pretrained_weights = MODELS[0]
        else:
            model_class, tokenizer_class, pretrained_weights = MODELS[model]

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)
        self.model = model_class.from_pretrained(pretrained_weights)

    def sent_to_idx(self, sent):
        ''' Converts a sentence to embeddings

            :param sent: the sentence to convert to embeddings
            :return: embeddings
        '''
        tokens = torch.tensor([self.tokenizer.encode("Let's see all hidden-states and attentions on this text")])
        with torch.no_grad():
            embs = self.model(tokens)[0]  # Models outputs are now tuples
       return embs



class SrcTrgLblData(Dataset):
    def __init__(self, src_filename, trg_filename, labels_filename):
        self.src_sents = open(src_filename).readlines()
        self.trg_sents = open(trg_filename).readlines()

        # in case we have test which means no labels
        self.labels = None
        if labels_filename is not None:
            self.labels = open(labels_filename).readlines()

        self.total_sents = len(self.src_sents)

    def __getitem__(self, idx):
        """Returns one data pair (source+<break>+target, label)."""
        src_sent_txt = self.src_sents[idx].strip()
        # tokenize
        # get embeddings

        trg_sent_txt = self.trg_sents[idx].strip()
        label_txt = '0' if self.labels is None else self.labels[idx].rstrip().lstrip()

        label = label_txt
        return src_sent_txt, trg_sent_txt, torch.tensor([float(label)], dtype=torch.float)

    def __len__(self):
        return self.total_sents

class DataLD(object):
    def __init__(self, src_path, trg_path, labels_path):
        self.dataset = SrcTrgLblData(src_path, trg_path, labels_path)

    def collate_fn(self, data):
        """ Creates mini-batch tensors from the list of tuples (src, trg, label).
            We should build a custom collate_fn rather than using default collate_fn,
            because merging sequences (including padding) is not supported in default.
            Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

            :params data: list of tuples (src_trg_combo, label).
            :returns: a torch tensor for the src_trg_combo, a torch tensor for the lables
                      and a list of length (batch_size).
        """
        def merge_vect(vectors):
            lengths = [len(vec) for vec in vectors]
            padded_vects = torch.zeros(len(vectors), max(lengths), dtype=torch.float32)

            for i, vec in enumerate(vectors):
                end = lengths[i]
                padded_vects[i, :end] = vec[:end]

            return padded_vects, lengths

        def merge_sent(sentences):
            lengths = [len(sent.split()) for sent in sentences]
            padded_sents = [['.' for _i in range(max(lengths))] for _j in range(len(sentences))]
            print(padded_sents)
            for i, sent in enumerate(sentences):
                end = lengths[i]
                padded_sents[i][:end] = sent.split()[:end]
            print(padded_sents)
            return [' '.join(psent) for psent in padded_sents], lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src, trg, labels = zip(*data)
        # merge sequences (from tuple of 1D tensor to 2D tensor)
        #src_sents, src_lengths = merge_sent(src)
        #trg_sents, trg_lengths = merge_sent(trg)
        src_lengths = [len(sent.split()) for sent in src]
        trg_lengths = [len(sent.split()) for sent in trg]

        labels, label_lengths = merge_vect(labels)
        return src, src_lengths, trg, trg_lengths, labels, label_lengths

    def get_loader(self, shuf=True, batch_size=100):
        """Returns data loader for custom dataset.

            :params batch_size: the batch size
            :returns: a data loader for the dataset.
        """
        # data loader for custome dataset
        # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
        # please see collate_fn for details
        data_loader = DataLoader(dataset=self.dataset,
                                 batch_size=batch_size,
                                 shuffle=shuf,
                                 collate_fn=self.collate_fn)

        return data_loader
