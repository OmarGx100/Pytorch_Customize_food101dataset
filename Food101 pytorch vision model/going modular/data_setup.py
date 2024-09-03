"""
Contains functionality for creating Pytorch Dataloader's for image classification data.
"""

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()

def create_dataloader(train_dir : str,
                        test_dir : str,
                        batch_size : int,
                        train_transform : transforms.Compose,
                        test_transform : transforms.Compose,
                        num_workers : int = NUM_WORKERS):
    
    """ Creates Training and testing dataloaders.

    Takes in a training and testing directory path and turns them into Pytorch Datasets,
    and then into Pytorch Dataloaders.
    The data should be in the form of 
    
    Data:
        |
        |
        |_____Train_data:
        |            |
        |            |__ class_one
        |            |            |
        |            |            |__ image_1.jpg
        |            |            |__ image_2.jpg
        |            |            |__ ...
        |            |
        |            |__ Class_two
        |            |            |__ image_1.jpg
        |            |            |__ image_2.jpg
        |            |            |__ ...
        |            |
        |            |__ .........
        |                        |__ img_1.jpg
        |                        |__ .....
        |                        |__ .....
        |_____Test_data:
                        |
                        |__ class_one
                        |            |
                        |            |__ image_1.jpg
                        |            |__ image_2.jpg
                        |            |__ ...
                        |
                        |__ Class_two
                        |            |__ image_1.jpg
                        |            |__ image_2.jpg
                        |            |__ ...
                        |
                        |__ .........
                                    |__ img_1.jpg
                                    |__ .....
                                    |__ .....

    Args : 
        train_dir : training data directory
        test_dir : testing data directory
        batch_size : how many examples are there in the batch
        transforms : preprocessing steps you might want to make on the data (i.e. transforms.ToTensor()) or some augmentation techniques
        num_workers : how many processcors you have access to, Default (os.cpu_count())

    Returns :
        A tuple of (train_dataloader, test_dataloader, class_names)
        where the class_names is the target class names

    Example Usage:
        train_dataloader, test_dataloader, class_names = create_dataloader(train_dir = path/to/training/data,
                                                                            test_dir = path/to/testing/data,
                                                                            batch_size = 32,
                                                                            train_transform = some_transformations,
                                                                            test_transform = some_transformations,
                                                                            num_workers = you can leave it with the default or you can set it to 1)
    """
    train_dataset = ImageFolder(train_dir,
                                transform = train_transform)

    test_dataset = ImageFolder(test_dir,
                                transform = test_transform)

    train_dataloader = DataLoader(train_dataset,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    num_workers = num_workers,
                                    pin_memory = True)
    test_dataloader = DataLoader(test_dataset,
    batch_size = batch_size,
    num_workers = num_workers,
    pin_memory = True)

    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names


