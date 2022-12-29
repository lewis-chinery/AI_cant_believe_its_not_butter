# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from data_preparation import get_dataloader
from CNN import CNN


def run_epoch(net, train_dl, val_dl):
    '''
    Runs one full epoch of training and validation
    
    :param net: CNN network that takes batches of images as input and outputs class predictions for each
    :param train_dl: dataloader for train dataset
    :param val_dl: dataloader for validation dataset
    
    :returns: two-element list of floats containing average train and validation loss for the epoch
    '''
    net.train()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    train_loss_record = []
    for batch_idx, (imgs, labels) in enumerate(train_dl):

        optimizer.zero_grad()
        
        predictions = net.forward(imgs)
        
        loss = criterion(predictions, labels)
        train_loss_record.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
    mean_val_loss = validate(net, val_dl)
        
    return np.mean(train_loss_record), mean_val_loss
        
        
def validate(net, val_dl):
    '''
    Calculate the Binary Cross Entropy loss on the validation dataset for a given CNN
    
    :param net: trained CNN
    :val_dl: dataloader for validation dataset
    
    :returns: float average validation loss
    '''
    net.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    val_loss_record = []
    for batch_idx, (imgs, labels) in enumerate(val_dl):
        
        predictions = net.forward(imgs)
        loss = criterion(predictions, labels)
        val_loss_record.append(loss.item())
        
    return np.mean(val_loss_record)

        
def train(net, train_dl, val_dl, epochs=10, patience=10):
    '''
    Train CNN for many epochs
    
    :param net: CNN network that takes batches of images as input and outputs class predictions for each
    :param train_dl: dataloader for train dataset
    :param val_dl: dataloader for validation dataset
    :param epochs: int number of epochs to train for
    :param patience: int number of epochs allowed without improvement of validation loss
    
    :returns: CNN than acheived lowest validation loss
              list of floats of average training loss for each epoch
              list of floats of average validation loss for each epoch
    '''
    train_loss_record = []
    val_loss_record = []
    for epoch in tqdm(range(epochs)):
        
        train_loss, val_loss = run_epoch(net, train_dl, val_dl)
        
        # save best net and early stopping
        if epoch == 0 or val_loss < np.min(val_loss_record):
            best_net = net
            epochs_without_improvement = 0
        elif epochs_without_improvement < patience:
            epochs_without_improvement += 1
        else:
            break
            
        train_loss_record.append(train_loss)
        val_loss_record.append(val_loss)
        
    return best_net, train_loss_record, val_loss_record


def evaluate(net, test_dl):
    '''
    Obtain predictions for a held out test set of images
    
    :param net: trained CNN
    :param test_dl: dataloader for test dataset
    
    :returns: ndarrays of test set labels and predictions
    '''
    for batch_idx, (imgs, labels) in enumerate(test_dl):
        
        predictions = net.forward(imgs)
        predictions = torch.sigmoid(predictions)
        
        try:
            labels_record += labels.numpy().tolist()
            predictions_record += predictions.detach().numpy().tolist()
        except NameError:
            labels_record = labels.numpy().tolist()
            predictions_record = predictions.detach().numpy().tolist()
            
    return np.array(labels_record).flatten(), np.array(predictions_record).flatten()
    
    
def main():
    '''
    Test training CNN on fake data
    '''
    # fake data
    n_train, n_val, n_test = 20, 10, 10
    
    X_train = [torch.rand(3, 150, 150) for n in range(n_train)]
    X_val   = [torch.rand(3, 150, 150) for n in range(n_val)]
    X_test  = [torch.rand(3, 150, 150) for n in range(n_test)]
    y_train = np.array([[int(np.round(np.random.rand(1)))] for n in range(n_train)])
    y_val   = np.array([[int(np.round(np.random.rand(1)))] for n in range(n_val)])
    y_test  = np.array([[int(np.round(np.random.rand(1)))] for n in range(n_test)])

    batch_size = 8
    train_dl = get_dataloader(X_train, y_train, batch_size=batch_size)
    val_dl   = get_dataloader(X_val,   y_val,   batch_size=batch_size)
    test_dl  = get_dataloader(X_test,  y_test,  batch_size=batch_size)
    
    # net
    net = CNN()
    
    # train
    epochs = 10
    print(f"\nTraining for {epochs} epochs")
    best_net, train_loss_record, val_loss_record = train(net, train_dl, val_dl, epochs=epochs)
    
    # evaluate
    print("\nTest set performance")
    labels, preds = evaluate(best_net, test_dl)
    df = pd.DataFrame(zip(labels, preds), columns=["labels", "preds"])
    print(df)
    
    
if __name__ == "__main__":
    
    main()
