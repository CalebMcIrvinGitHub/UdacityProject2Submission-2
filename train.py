import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np

import PIL

import json

import argparse

data_dir=sys.argv[1]

parser = argparse.ArgumentParser()

parser.add_argument("--save_dir", dest='save_dir')
parser.add_argument("--arch", dest='arch')
parser.add_argument("--hidden_units", dest='hidden_units')
parser.add_argument("--gpu", dest='gpu', nargs='*')
parser.add_argument("--learning_rate", dest='learning_rate')
parser.add_argument("--epochs", dest='epochs')

args, unknown = parser.parse_known_args()

print(data_dir)
print(args.save_dir)
print(args.arch)
print(args.hidden_units)
print(args.gpu)
print(args.learning_rate)
print(args.epochs)

checkpoint_dir=args.save_dir

if not args.arch == None:
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        model=models.resnet18(pretrained=True)
else:
    model = models.resnet50(pretrained=True)
    print("No Architecture Specified")
    
if not args.gpu == None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if not args.learning_rate == None:
    learning_rate = float(args.learning_rate)
else:
    learning_rate = .001
    
if not args.epochs == None:
    epochs = int(args.epochs)
else:
    epochs = 4

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_training_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((.485, .456, .406), (.229, .224, .225))])

data_valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((.485, .456, .406), (.229, .224, .225))])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=data_training_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=data_valid_test_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=data_valid_test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)  


model = models.resnet50(pretrained=True)
model.to(device)

for param in model.parameters():
    param.requires_grad=False

model.fc = nn.Sequential(nn.Linear(2048, int(args.hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(args.hidden_units), 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

model.to(device)

steps = 0
running_loss = 0
print_every = 50
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

running_loss = 0
batch_loss = 0
            
for epoch in range(epochs):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    running_loss = 0

checkpoint = {'classifier': model.fc,
              'learning_rate': learning_rate,
              'state_dict': model.state_dict(),
              'class_to_idx': train_datasets.class_to_idx,
              'optimizer_dict': optimizer.state_dict()}
torch.save(checkpoint, checkpoint_dir)
