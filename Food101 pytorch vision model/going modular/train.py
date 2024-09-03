import pathlib
import os

import torch
from torchvision import transforms
from engin import train
from utils import save_model_to_dir
from model_builder import BaseLineModel
from data_setup import create_dataloader
from helper_functions import accuracy_fn
"""
    Python script to train our tinyVGG model with device agnostic code using command line 
"""

# SETTING SOME HYPERPARAMETERS
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_UNITS = 10
INPUTS_CHANNELS = 3

# SETTING DIRECTORIES
train_data_dir = "./../Data/pizza_steak_sushi/train"
test_data_dir = "./../Data/pizza_steak_sushi/test"

# Setting model name and saving directory
model_name = "saved_from_train_py.pth"
model_dir = "./../Models"

# SETTING DEVICE AGNOSTIC CODE
device = "cuda" if torch.cuda.is_available() else "cpu"

# CREATE TRANSFORMS 
train_data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
test_data_transforms  = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Creating training and testing dataloader using our create dataloader function from data_setup.py
train_dataloader, test_dataloader ,class_names = create_dataloader(train_dir=train_data_dir,
                                                                   test_dir=test_data_dir,
                                                                   batch_size = BATCH_SIZE,
                                                                   train_transform=train_data_transforms,
                                                                   test_transform= test_data_transforms,
                                                                   num_workers= 1)

model = BaseLineModel(input_shape = INPUTS_CHANNELS,
                      hidden_units= HIDDEN_UNITS,
                      output_shape = len(class_names))

optimizer = torch.optim.Adam(params = model.parameters(),
                             lr = LEARNING_RATE)

loss_fnc = torch.nn.CrossEntropyLoss()

accuracy_fnc = accuracy_fn

if __name__ =="__main__":
    resutls = train(model = model,
                    train_data_loader=train_dataloader,
                    test_data_loader = test_dataloader,
                    optimzer=optimizer,
                    loss_fnc= loss_fnc,
                    device = device,
                    epochs = NUM_EPOCHS,
                    acc_fnc= accuracy_fn,
                    early_stop = True)

    save_model_to_dir(model_name, model_dir, model.state_dict())
