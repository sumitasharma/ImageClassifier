import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F

from collections import OrderedDict
from PIL import Image
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

models_dict = {'vgg16': models.vgg16(pretrained = True),
               'resnet18': models.resnet18(pretrained = True),
               'alexnet': models.alexnet(pretrained = True)
              }
# Parses command line arguments and returns a dictionary of parsed arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Parser for Image Classifier Training")
    parser.add_argument("--data_directory", type=str, default="./data")   
    parser.add_argument("--save_dir", type=str , default="./models")
    parser.add_argument("--arch", type=str, default="vgg16")
    parser.add_argument("--learning_rate", type=float, default=.00005)
    parser.add_argument("--hidden_units", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    print(parser.parse_args())
    args_dict = vars(parser.parse_args())
    return args_dict

# Transform input data and return data loaders for training
def prepare_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                         ])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                           ])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                        ])


    training_datasets = datasets.ImageFolder(train_dir , transform = training_transforms) 
    validation_datasets = datasets.ImageFolder(valid_dir , transform = validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir , transform = testing_transforms)

    trainloaders = torch.utils.data.DataLoader(training_datasets , batch_size = 64, shuffle = True )
    validationloaders = torch.utils.data.DataLoader(validation_datasets , batch_size = 32 ) 
    testloaders = torch.utils.data.DataLoader(testing_datasets , batch_size = 32 ) 
    
    return (trainloaders, testloaders, validationloaders, training_datasets) 

def build_model(arch, hidden_units):
    print('arch is', arch)
    print('models_dict is', models_dict)
    
    if arch not in models_dict:
         print('Invalid value for \'arch\' parameter. \'arch\' parameter should be one of (vgg16, alexnet or resnet18)')
         return None
    
    model = models_dict[arch]
    # Freeze parameters and swap the classifier with own classifier
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim =1))]))
    model.classifier = classifier
    return model

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train_model(model, trainloader, validationloaders, epochs, print_every, learning_rate, device='cpu'):

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    epochs = epochs
    print_every = print_every
    steps = 0
    running_loss = 0

    # change to cuda
    model.to(device)

    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validationloaders, criterion, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validationloaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloaders)))

                running_loss = 0
    return model , optimizer
    
def check_accuracy_on_test(model, testloader, device = 'cpu'):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def save_model(model, optimizer, training_datasets, save_dir):
    model.class_to_idx = training_datasets.class_to_idx
    checkpoint = {'arch':'vgg16',
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier' : model.classifier
                 }
    torch.save(checkpoint, save_dir +'/checkpoint')
        
def main():
    args_dict = parse_args()
    print(args_dict)
    data_dir = args_dict['data_directory']

    # Sets up datasets and dataloaders for training
    (train_dataloader, test_dataloader, validation_dataloader, training_datasets) = prepare_data(data_dir)
    
    # Build Model
    model = build_model(args_dict['arch'],
                        args_dict['hidden_units'])
    if args_dict['gpu']:
        device_type = 'cuda'
        print('Running on GPU')
    else:
        print('Running on CPU')
        device_type = 'cpu'
    model ,optimizer = train_model(model, train_dataloader, validation_dataloader, args_dict['epochs'], 40, args_dict['learning_rate'], device_type)
    check_accuracy_on_test(model, test_dataloader, device_type)
    save_model(model, optimizer, training_datasets, args_dict['save_dir'])

    
if __name__ == "__main__":
    main()