# https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid


def plot_loss(train_loss_record, val_loss_record):
    '''
    Plot train and validation loss
    
    :param train_loss_record: list of floats of average training loss for each epoch
    :param val_loss_record: list of floats of average validation loss for each epoch
    '''
    plt.plot(train_loss_record, label="train")
    plt.plot(val_loss_record, label="val")
    
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.show()
    
    
def plot_predictions_histogram(labels, predictions, class_to_idx, bins=20, binrange=(0,1)):
    '''
    Plot overlapping histogram showing how well the CNN predictions match the true labels
    
    :param labels: list of ints of image labels
    :param predictions: list of floats of image predictions
    :param class_to_idx: dictionary mapping the integer labels to descriptive str labels
    :param bins: int num bins on hist
    :param binrange: tuple x min and max of hist
    '''
    df = pd.DataFrame(zip(labels, predictions), columns=["labels", "preds"])
    
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    df["labels"] = df["labels"].astype(int)
    df["labels"] = df.apply(lambda row: f'{row["labels"].astype(int)} - {idx_to_class[row["labels"]]}', axis=1)
    
    sns.histplot(df, x="preds", hue="labels", bins=bins, binrange=binrange)
    plt.title("'Spread' of predictions")
    plt.show()
        

def plot_image_examples(X, y, cls, class_to_idx, max_images=None, nrow=6, figsize=(12, 6)):
    '''
    Plot example images belonging to a certain class from e.g. the train dataset
    
    :param X: tensor [batch_size, 3, 150, 150] of images
    :param y: tensor [batch_size, 1] of labels
    :param class_to_idx: dictionary mapping the integer labels to descriptive str labels
    :param max_images: int max number of images to show
    :param nrow: int max number of images to display on one row
    :param figsize: tuple max figure size
    '''
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    X_cls = [img for idx, img in enumerate(X) if idx_to_class[y[idx][0]]==cls]
    X_cls = X_cls[:max_images] if max_images else X_cls
    
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f"{cls.capitalize()}", fontsize=16)
    plt.imshow(make_grid(X_cls, nrow=nrow).permute(1,2,0))
    plt.show()
