# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import codecs
import time
import argparse

import NN.lstm_classifier_minibatch as lstm_classifier_minibatch
import utils.dictionary as Dict
import utils.data_preprocess as DP
import utils.visualise as VIS

import os
import sys
import random

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

def train(train_data, dev_data, test_data, batch_size=100, device=torch.device("cpu"), attention=None, models_dir='models'):
    """ training the model one epoch at a time

    :param training_data: the training data (data_preprocess type)
    :param dev_data: the development data (data_preprocess type)
    :param batch_size: the size of the minibatches (default 64)
    :param device: the device to be used (default "cpu")
    :param attention: the type of attention ('dot', 'rte', 'None')
    :param models_dir: the directory to store models in
    """
    EMBEDDING_DIM = 768
    HIDDEN_DIM = 64
    EPOCH = 100
    BATCH_SIZE = batch_size
    METRIC = 'eucledian'
    ATTENTION = attention

    model = lstm_classifier_minibatch.SiameseSimilarity(embedding_dim=EMBEDDING_DIM,
                                                     hidden_dim=HIDDEN_DIM,
                                                     dict_size=train_data.get_dict_size(),
                                                     batch_size=BATCH_SIZE,
                                                     metric = METRIC,
                                                     attention_type= ATTENTION,
                                                     device = device)

    model.to(device)

    # Uncomment/comment the loss function you want
#    loss_function = lstm_classifier_minibatch.ContrastiveLoss()
    loss_function = nn.MSELoss()
#    optimizer = optim.SGD(model.parameters(), lr = 1e-2)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    updates = 0
    loss_history = []
    best_dev_pearson = -1.0
    best_epoch = -1

    for epoch in range(EPOCH):
        start_time = time.time()

        print('\nStart epoch: %d with Hyperparams: %s ' % (epoch, ATTENTION))
        loss_float = train_epoch(model, train_data, loss_function, optimizer, batch_size, epoch, device)
        loss_history.append(loss_float)

        elapsed_time = time.time() - start_time
        print('Epoch elapsed time: ' + str(elapsed_time) + "\n")

        dev_pearson, dev_mae, dev_rmse, dev_pred = evaluate(model, dev_data, loss_function, 'dev', device)
        test_pearson, test_mae, test_rmse, test_pred = evaluate(model, test_data, loss_function, 'test', device)

        if best_dev_pearson > -1.0:
            print('Best dev pearson: %.2f in epoch %d' % (best_dev_pearson, best_epoch))

        if dev_pearson > best_dev_pearson:
            best_dev_pearson = dev_pearson
            best_epoch = epoch
            if os.path.exists(models_dir):
                if os.listdir(models_dir):
                    cleanup_best_command = "rm " + os.path.join(models_dir, "best_model_minibatch_*.model")
                    os.system(cleanup_best_command)
            else:
                print('Creating directory ' + os.path.realpath(models_dir))
                os.mkdir(models_dir)

            test_pred = test(model, test_data, device)

            print('New Best Dev!!!')
            torch.save(model, os.path.join(models_dir, "best_model_minibatch_" + str(int(dev_pearson*10000)) + "_" + str(epoch) + '.model'))

            with open(os.path.join(models_dir, "best_scores.dev"), "w") as oF:
                oF.write('\n'.join([str(float(d)) for d in dev_pred]))

            with open(os.path.join(models_dir, "best_metrics.dev"), "w") as oF:
                oF.write(' '.join([str(dev_pearson), str(dev_mae), str(dev_rmse)]))

            with open(os.path.join(models_dir, "best_scores.test"), "w") as oF:
                oF.write('\n'.join([str(float(d)) for d in test_pred]))

            updates = 0
        else:
            updates += 1
            if updates >= 10:
                break
            torch.save(model, os.path.join(models_dir, "model_minibatch_" + str(int(dev_pearson*10000)) + "_" + str(epoch) + '.model'))

    #VIS.show_plot(list(range(epoch+1)),loss_history)

def train_epoch(model, train_data, loss_function, optimizer, batch_size, epoch, device=torch.device("cpu")):
    """ training the model during an epoch

    :param train_data: an iterator over the training data.
    :param loss_function: the loss function.
    :param optimizer: the learning optimizer.
    :param batch_size: the size of the batch
    :param epoch: the current epoch count
    :param device: the device to be used (default cpu)
    :returns: the loss
    """

    #enable training mode
    model.train()

    avg_loss = 0.0
    avg_acc = 0.0
    total = 0
    total_acc = 0.0
    total_loss = 0.0

    for iter, traindata in enumerate(train_data.get_loader(shuf=True, batch_size=batch_size)):
        train_src, _train_src_length, train_trg, _train_trg_length, train_labels, _train_labels_length = traindata
        train_labels = torch.squeeze(train_labels)

        train_src = train_src.to(device)
        train_trg = train_trg.to(device)
        train_labels = train_labels.to(device)

        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        output = model(train_src.t(), train_trg.t())
        (output_src, output_trg, sim) = output

        loss = loss_function(sim, train_labels)
        loss.backward()
        optimizer.step()

    print("Current loss: " + str(loss.item()) + "\n")
    return loss.item()

def evaluate(model, data, loss_function, name ='dev', device=torch.device("cpu")):
    """ evaluate a model.

    :param model: the NMT model
    :param data: the data that we want to evaluate
    :param loss_function: the loss_function
    :param vocabulary_to_idx: the dictionary of words to indexes
    :param label_to_idx: the dictionary of labels to indexes
    :param name: the name of the test
    :param device: the device to be used (default cpu)
    :returns: the divergence from the loss
    """
    model.eval()
    rmse = 0.0
    mae = 0.0
    avg_loss = 0.0
    total = 0

    sim_list = []
    test_label_list = []

    start_time = time.time()
    for iter, testdata in enumerate(data.get_loader(shuf=False, batch_size=1)):
        test_src, _test_src_length, test_trg, _test_trg_length, test_label, _test_labels_length = testdata
        test_label = torch.squeeze(test_label, 0)

        model.batch_size = 1

        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        test_src = test_src.to(device)
        test_trg = test_trg.to(device)
        test_label = test_label.to(device)

        prediction = model(test_src.t(), test_trg.t())
        (prediction_src, prediction_trg, sim) = prediction

        loss = loss_function(sim, test_label)

        #print('sim: ' + str(sim.shape) + " " + str(sim.item()))
        #print('test: ' + str(test_label.shape) + " " + str(test_label.item()))

        sim_list.append(sim[0].item())
        rmse += ((sim[0].item() - test_label.item())**2)
        mae += np.abs(sim[0].item() - test_label.item())

        test_label_list.append(test_label.item())
        total += 1
        avg_loss += loss.item()

    elapsed_time = time.time() - start_time
    #VIS.save_plot2(sim_list, test_label_list, os.path.join(os.getcwd(), 'plots', str(round(time.time())) + '.png'))
    #VIS.export_values(sim_list, name)
    avg_loss /= total
    rmse = (rmse / total)**0.5
    mae /= total
    pearson = np.corrcoef(sim_list, test_label_list)[1,0] if not np.isnan(np.corrcoef(sim_list, test_label_list)[1,0]) else 0.0

    print(name + ' avg_loss:%g pearson:%g mae:%g rmse:%g' % (avg_loss, pearson, mae, rmse))
    print(name + ' elapsed time: ' + str(elapsed_time))
    return pearson, mae, rmse, sim_list

def test(model, data, device=torch.device("cpu")):
    """ evaluate a model.

        :param model: the NMT model
        :param data: the data that we want to evaluate
        :param device: the device to be used (default cpu)

        :returns: the divergence from the loss
    """
    model.eval()

    sim_list = []

    start_time = time.time()
    for iter, testdata in enumerate(data.get_loader(shuf=False, batch_size=1)):
        test_src, _test_src_length, test_trg, _test_trg_length, test_label, _test_labels_length = testdata
        test_label = torch.squeeze(test_label)

        model.batch_size = 1

        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        test_src = test_src.to(device)
        test_trg = test_trg.to(device)

        prediction = model(test_src.t(), test_trg.t())
        (prediction_src, prediction_trg, sim) = prediction

        sim_list.append(sim[0].item())

    elapsed_time = time.time() - start_time

    return sim_list


def main():
    ''' read arguments from the command line and initiate the training.
    '''

    parser = argparse.ArgumentParser(description='Train an LSTM sentence-pair classifier.')
    parser.add_argument('-d', '--data-folder', required=True, help='the folder containing the train, test, dev sets.')
    parser.add_argument('-s', '--source-ext', required=False, default='src', help='the extension of the source files.')
    parser.add_argument('-t', '--target-ext', required=False,  default='mt', help='the extension of the target files.')
    parser.add_argument('-l', '--labels-ext', required=False,  default='hter', help='the extension of the labels files.')
    parser.add_argument('-b', '--batch-size', required=False, default=64, help='the batch size.')
    parser.add_argument('-a', '--attention-type', required=False, default=None, help='the attention type: \'dot\', \'rte\', \'None\'.')
    parser.add_argument('-m', '--model-folder', required=False, default='models', help='the directory to save the models')
    parser.add_argument('-g', '--gpuid', required=False, default=-1, help='the ID of the GPU to use.')

    args = parser.parse_args()

    source_train_filename = os.path.join(os.path.realpath(args.data_folder), 'train.' + args.source_ext)
    target_train_filename = os.path.join(os.path.realpath(args.data_folder), 'train.' + args.target_ext)
    source_dev_filename = os.path.join(os.path.realpath(args.data_folder), 'dev.' + args.source_ext)
    target_dev_filename = os.path.join(os.path.realpath(args.data_folder), 'dev.' + args.target_ext)
    source_test_filename = os.path.join(os.path.realpath(args.data_folder), 'test.' + args.source_ext)
    target_test_filename = os.path.join(os.path.realpath(args.data_folder), 'test.' + args.target_ext)

    labels_train = os.path.join(os.path.realpath(args.data_folder), 'train.' + args.labels_ext)
    labels_dev = os.path.join(os.path.realpath(args.data_folder), 'dev.' + args.labels_ext)

    labels_test = os.path.join(os.path.realpath(args.data_folder), 'test.' + args.labels_ext)
    if not os.path.exists(labels_test):
        labels_test = None

    data_dict_file = os.path.join(os.path.realpath(args.data_folder), 'data.dict')
    #labels_dict_file = os.path.join(os.path.realpath(args.data_folder), 'ter.dict')
    dictionary_data = Dict.Dictionary(data_dict_file)
    dictionary_data.load_dictionary()
    #dictionary_labels = Dict.Labels(labels_dict_file)
    #dictionary_labels.load_dictionary()

    train_data = DP.DataLD(source_train_filename, target_train_filename, labels_train, dictionary_data) #, dictionary_labels)
    dev_data = DP.DataLD(source_dev_filename, target_dev_filename, labels_dev, dictionary_data) #, dictionary_labels)
    test_data = DP.DataLD(source_test_filename, target_test_filename, labels_test, dictionary_data) #, dictionary_labels)

    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() and int(args.gpuid) > -1 else "cpu")

    train(train_data, dev_data, test_data, int(args.batch_size), device, args.attention_type, args.model_folder)

if __name__ == "__main__":
    main()
