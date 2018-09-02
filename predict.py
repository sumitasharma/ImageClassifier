# Command to Run
# python predict.py --gpu --input="./data/test/101/image_07949.jpg" --> trumpet creeper
# python predict.py --gpu --input="./data/test/10/image_07090.jpg" --> globe thistle
# python predict.py --gpu --input="./data/test/1/image_06752.jpg" --> pink_primrose
# python predict.py --gpu --input="./data/test/13/image_05761.jpg" --> king protea

import argparse
import json
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

# Parses command line arguments and returns a dictionary of parsed arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Parser for Image Classifier Training")
    parser.add_argument("--data_directory", type=str, default="./data")   
    parser.add_argument("--save_dir", type=str , default="./models")
    parser.add_argument("--top_k", type=int , default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--input", type=str, default="./data/test/1/image_06752.jpg")
    print(parser.parse_args())
    args_dict = vars(parser.parse_args())
    return args_dict
    
def load_model(save_dir, gpu):
    model = models.vgg16()
   
    if gpu:
        model.cuda()
    checkpoint = torch.load(save_dir + '/checkpoint')
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    print(model)
    return model

def process_image(image):
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pil_image.thumbnail((256,256))
    pil_image = pil_image.resize((224,224))
    np_image = np.array(pil_image) / 255
    np_mean = np.array([0.485, 0.456, 0.406])
    np_std  = np.array([0.229, 0.224, 0.225])    
    np_image = (np_image - np_mean)/np_std
    np_image = np.transpose(np_image, (2 ,0 ,1))  
    return np_image

def predict(image_path, model, topk=5):

    # TODO: Implement the code to predict the class from an image file
    np_image = process_image(image_path)
    img = torch.FloatTensor(np_image).cuda()
    img.unsqueeze_(0)
    
    output = model.forward(img)

    ps = F.softmax(output, dim=1)
    print(torch.topk(ps, topk))
    return torch.topk(ps, topk)

def main():
    args_dict = parse_args()
    print(args_dict)
    model = load_model(args_dict['save_dir'], args_dict['gpu'])
    probs, classes = predict(args_dict['input'], model, args_dict['top_k'])
    
    classes_numpy = classes.cpu().numpy()
    probs_numpy = (probs.data).cpu().numpy()
    
    print('classes_numpy is', classes_numpy)
    print('probs_numpy is', probs_numpy)
    print('model.class_to_idx is' , model.class_to_idx)
#    idx = []
#     for key in classes_numpy[0]:
#         # print(key)
#         # print(model.class_to_idx[key])
#         idx.append(model.class_to_idx[str(key)])
#     print (idx)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        print('cat_to_name is', cat_to_name)
    name_to_prob = {}
    for index,value in enumerate(classes_numpy[0]):
        print('index,value is', index, value)
        for key in model.class_to_idx:
            #print('key is', key)
            if (model.class_to_idx[key] == value):
                name_to_prob[probs_numpy[0][index]] = cat_to_name[str(key)]
    
    print(name_to_prob)

if __name__ == "__main__":
    main()