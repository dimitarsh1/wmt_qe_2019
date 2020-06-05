import os
import torch
#import torch.autograd as autograd
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

class DataPreprocessor(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.dict_length = len(dictionary)

    def tokenise(self, sent):
        """ splits a sentence into a list of tokens
        
        :param sent: the sentence in a text form.
        :returns: a list of tokens
        """
        return sent.strip().split() # very basic for now.
    
    def sentlist_to_idx(self, sent_list, pre_char='<sos>', suf_char='<eos>', break_char='<break>'):
        """ Converts a list of sentences to a list of indexes, concatenating
            all sentences using the break_char inbetween
        
        :param sent_list: a list of sentences in text format
        :returns: a list of indexes
        """
        if break_char != " ":
            break_char = " " + break_char + " "
            
        sent_combo = " ".join([pre_char, break_char.join(sent_list), suf_char])
        tokens = self.tokenise(sent_combo)
        idxs = [self.dictionary.get_idx(token) for token in tokens]
        return idxs

    def make_one_hot(self, idx):
        ''' Creates a one-hot representation of the index value

            :param idx: an int value to convert to a one-hot encoded vector of shape (dict_length, 1)
            :returns: one-hot encoding of the index
        '''
        one_hot = torch.zeros(self.dict_length, 1)
        one_hot[idx] = 1.0

        return one_hot
        
    def sent_to_idx(self, sent, pre_char='<sos>', suf_char='<eos>'):
        tokens = self.tokenise(" ".join([pre_char, sent, suf_char]))
        idxs = [self.dictionary.get_idx(token) for token in tokens]
        return idxs

    def sent_to_onehot(self, sent,  pre_char='<sos>', suf_char='<eos>'):
        tokens = self.tokenise(" ".join([pre_char, sent, suf_char]))
        idxs = [self.make_one_hot(self.dictionary.get_idx(token)) for token in tokens]
        return idxs


class SrcTrgLblData(Dataset):
    def __init__(self, src_filename, trg_filename, labels_filename, dictionary_data):
        self.src_sents = open(src_filename).readlines()
        self.trg_sents = open(trg_filename).readlines()

        # in case we have test which means no labels
        self.labels = None
        if labels_filename is not None:
            self.labels = open(labels_filename).readlines()

        self.total_sents = len(self.src_sents)
        self.preprocessor_data = DataPreprocessor(dictionary_data)

    def __getitem__(self, idx):
        """Returns one data pair (source+<break>+target, label)."""
        src_sent_txt = self.src_sents[idx]
        trg_sent_txt = self.trg_sents[idx]
        label_txt = '0' if self.labels is None else self.labels[idx].rstrip().lstrip()

        #src_idx = self.preprocessor_data.sent_to_onehot(src_sent_txt)
        #trg_idx = self.preprocessor_data.sent_to_onehot(trg_sent_txt)
        src_idx = self.preprocessor_data.sent_to_idx(src_sent_txt)
        trg_idx = self.preprocessor_data.sent_to_idx(trg_sent_txt)
        # label_idx = self.dictionary_lbl.get_idx(label_txt.strip())
        # label = self.dictionary_lbl.get_token(label_idx)
        label = label_txt
        return torch.tensor(src_idx, dtype=torch.long), torch.tensor(trg_idx, dtype=torch.long), torch.tensor([float(label)], dtype=torch.float)
        #return src_idx, trg_idx, torch.FloatTensor([float(label)])

    def __len__(self):
        return self.total_sents

class DataLD(object):
    def __init__(self, src_path, trg_path, labels_path, dictionary_data):
        self.data_dict_size = len(dictionary_data)
        self.dataset = SrcTrgLblData(src_path, trg_path, labels_path, dictionary_data)

    def collate_fn(self, data):
        """ Creates mini-batch tensors from the list of tuples (src_trg_combo, label).
            We should build a custom collate_fn rather than using default collate_fn,
            because merging sequences (including padding) is not supported in default.
            Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
            
            :params data: list of tuples (src_trg_combo, label).
            :returns: a torch tensor for the src_trg_combo, a torch tensor for the lables
                      and a list of length (batch_size).
        """
        def merge(sequences, type='long'):
            lengths = [len(seq) for seq in sequences]
            if type == 'float':
                padded_seqs = torch.zeros(len(sequences), max(lengths), dtype=torch.float32)
            else:
                padded_seqs = torch.zeros(len(sequences), max(lengths), dtype=torch.int64)

            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]

            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x[0]), reverse=True)

        # seperate source and target sequences
        src, trg, labels = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_sents, src_lengths = merge(src)
        trg_sents, trg_lengths = merge(trg)
        labels, label_lengths = merge(labels, type='float')
        return src_sents, src_lengths, trg_sents, trg_lengths, labels, label_lengths

    def get_dict_size(self):
        """ returns the size of the data dictionary """
        return self.data_dict_size
        
#    def get_labels_size(self):
#        """ returns the size of the labels dictionary """
#        return self.labels_dict_size
        
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
