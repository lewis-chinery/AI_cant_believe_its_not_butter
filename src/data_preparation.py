# https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
# https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00


import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


class ButterDataset(Dataset):
    '''
    Custom dataset to process data for input to dataloader
    X and y are outputs of split_dataset_into_train_val_test()
    '''
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype(float)  # float casting needed for bceloss
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]
        return img, label
    

def delete_ipynbcheckpoints(directory):
    '''
    Delete .ipynb_checkpoints from a directory
    Jupyter likes creating these checkpoints but ImageFolder is not a fan
    
    :param directory: str full path to directory possibly containing .ipynb_checkpoints
    '''
    ipynb_dir = os.path.join(directory, ".ipynb_checkpoints")
    if os.path.exists(ipynb_dir):
        shutil.rmtree(ipynb_dir)


def load_data(data_dir):
    '''
    Loads images from data_dir directory, processes them so there are all 150x150 pixels, and saves them as a dataset object
    
    :param data_dir: str full path to data directory containing sub dirs with images for each class separately
    
    :returns: dataset object with 3 channel tensor of RGB image plus integer class label
              integer-class relationship can be found using dataset.class_to_idx
    '''
    delete_ipynbcheckpoints(data_dir)
    dataset = ImageFolder(data_dir,
                          transform = transforms.Compose([transforms.Resize((150,150)), transforms.ToTensor()])
                         )    
    return dataset


def split_dataset_into_train_val_test(dataset, val_pc, test_pc, seed=0):
    '''
    Splits dataset object containing all classes into stratified train, val, and test sets
    
    :param dataset: Dataset object containing images and labels for all classes
    :param val_pc: float between 0 and 1 for val set proportion
    :param test_pc: float between 0 and 1 for test set proportion
    :param seed: num to fix train_test_split seed for reproducibility
    
    :returns: list of 3x150x150 tensors for X_train etc
              list of (1,) shape ndarrays for y_train etc
    '''
    # adjust val pc to be proportion less test set
    val_pc = val_pc / (1-test_pc)
    
    X = [img for (img, label) in dataset]
    y = [label for (img, label) in dataset]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pc, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pc, random_state=seed, stratify=y_train_val)
    
    return X_train, X_val, X_test, np.expand_dims(y_train, -1), np.expand_dims(y_val, -1), np.expand_dims(y_test, -1)


def get_dataloader(X, y, batch_size=8, shuffle=False):
    '''
    Creates dataloader object for use during training
    
    :param X: list of 3x150x150 tensors i.e. RGB images
    :param y: list of (1,) shape ndarrays i.e. labels
    :batch_size: int size of batches used by dataloader
    :shuffle: bool shuffle dataloader output
    
    :returns: Dataloader object containing 3 channel images and labels
              next(iter(dataloader)) will return 2 element list of tensors
              first element: images of size [batch_size, 3, 150, 150]
              second element: labels of size [batch_size, 1]
    '''
    dataset = ButterDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    
def main():
    '''
    Test data preparation code and examine dimensions of outputs
    '''
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(root_dir, "data")
    
    dataset = load_data(data_dir)
    
    val_pc, test_pc, batch_size = 0.2, 0.2, 8
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_into_train_val_test(dataset, val_pc, test_pc)

    print(f"\nTrain set size:\t{len(X_train)}")
    print(f"Val set size:\t{len(X_val)}")
    print(f"Test set size:\t{len(X_test)}")
    print(f"\nImage data shape: \t{X_test[0].shape}\n")
    

if __name__ == "__main__":
    
    main()
