# AI Programming with Nanodegree Image Classifier Project
## Overview
This project is part of the AI Programming with Nanodegree Image Classifier Project. The goal of the project is train on sample images for different category of flowers and then given an image predict the category of a flower.

It has the following files:
1. train.py - This file provides the code to prepare data, build and train the model.
2. predict.py - This file provides the code to predict an image file.
3. Image Classifier Project.ipynb - Jupyter notebook for the project.
4. cat_to_name.json - A file with a mapping from predicted class to name

data folder - Users should create a folder named 'data' and download/unzip training/testing/validation data from [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) in the folder.
models folder - Users should create a folder named 'models' to store the trained models that can be loaded for prediction

## Running the project
The project can be run using the following commands:

### Training the model
``` 
python train.py --gpu 
```

### Predicting Images
```
python predict.py --gpu --input="./data/test/101/image_07949.jpg" 
```
This should output 'trumpet creeper'
