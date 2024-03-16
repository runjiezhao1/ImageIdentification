# Image Identification Using CNN

## Description
Given the image, we will give it one of ten labels based on the content in the image.
Labels: airplane, automobile, bird, cat, deer, dog, frog, horse

## Data
Image data are all from CIFAR-10.

## Basic Model
The accuracy rate of ```Net``` model in the ```models.py``` is 58%.<br />
The accuracy rate of ```ImprovedNet``` model in the ```models.py``` is 64%.

## Preprocess
whitening method: xij = (pij - mean) / std<br />
result = (b - a)*(xij - min) / (max - min)