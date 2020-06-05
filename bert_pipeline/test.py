# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(8)
torch.manual_seed(1)

import codecs 

import NN.lstm_classifier_minibatch as lstm_classifier_minibatch

import argparse
import os
import utils.dictionary as Dict
from utils.data_preprocess import DataPreprocessor


def test(src_sent, trg_sent, data_pr, model_path, deviceid=-1):
    """ Tests a sentence pair whether they match or not
    
    :param src_sent: the source sentence
    :param trg_sent: the target sentence
    :param data_pr: the data preprocessor that contains the dictionary and the preprocssing methods
    :param model_path: the path to the model (saved during training)
    :param deviceid: the id of the device to run all computations
    :returns: the best class
    """
    
    if deviceid > -1:
        torch.cuda.set_device(deviceid)
        
    if deviceid > -1:
        model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(deviceid))
        model.cuda()
    else:
        model = torch.load(model_path, map_location='cpu')
        
    model.eval()
    model.hidden = model.init_hidden()
    
    test_src = data_pr.sent_to_idx(src_sent)
    test_trg = data_pr.sent_to_idx(trg_sent)

    test_src_tensor = torch.LongTensor(test_src)
    test_trg_tensor = torch.LongTensor(test_trg)        
    
    if deviceid > -1 :
        test_src_var = autograd.Variable(test_src_tensor.cuda())
        test_trg_var = autograd.Variable(test_trg_tensor.cuda())
    else:
        test_src_var = autograd.Variable(test_src_tensor)
        test_trg_var = autograd.Variable(test_trg_tensor)
        
    prediction = model(test_src_var, test_trg_var)
    (prediction_src, prediction_trg, sim) = prediction

    pred = model.compute_class_prediction(sim.data)
    return pred.numpy()[0] #(sim.data[0]).numpy()[0]
    
def main():
    parser = argparse.ArgumentParser(description='Clasifying a sentence with LSTM.')
    parser.add_argument('-m', '--model', required=True, help='the path to the model file.')
    parser.add_argument('-s', '--source-sentence', required=True, help='the source sentence to classify')
    parser.add_argument('-t', '--target-sentence', required=True, help='the target sentence to classify')
    parser.add_argument('-d', '--dictionary', required=True, help='the dictionary/vocabulary.')
    parser.add_argument('-g', '--gpuid', required=False, default=-1, help='the ID of the GPU to use.')
    
    args = parser.parse_args()
    
    deviceid = -1
    if int(args.gpuid) > -1 and torch.cuda.is_available():
        deviceid = int(args.gpuid)
        print('Using GPU ' + str(deviceid))
        torch.cuda.set_device(deviceid)
        
    if os.path.exists(args.dictionary):
        dictionary = Dict.Dictionary(os.path.realpath(args.dictionary))
        dictionary.load_dictionary()
        data_pr = DataPreprocessor(dictionary)
    else:
        print("Path not found: ", args.dictionary)
        exit(1)
    
    if not os.path.exists(args.model):
        print("Path not found: ", args.model)
        exit(1)    

    print(test(args.source_sentence, args.target_sentence, data_pr, args.model, deviceid))
        
if __name__ == "__main__":
    main()
    
