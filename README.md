# Convolutional Neural Networks for P300 Detection
This repository evaluates different state-of-the-art CNN arquitectures for P300 detection in EEG using the P300 Speller dataset. 

## Requirements
* Python 3.7
* Tensorflow 1.12.0
* NumPy
* matplotlib
* scikit-learn
* cudatoolkit 10.0
* cudnn

You can create an [conda environment](https://www.anaconda.com/distribution/) with all the dependencies using the `environment.yml` file in this repository.

```
conda env create -n p300cnn -f environment.yml
```