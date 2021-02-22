# Convolutional Neural Networks for P300 Detection
This repository evaluates different state-of-the-art CNN arquitectures for P300 detection in EEG signals and compares them in terms of detection performance and model complexity. 

## Datasets
The evaluation was done on the following datasets:

* [P300 Akimpech Database](https://akimpech.izt.uam.mx/p300db/p300db.html) ([LINI](https://akimpech.izt.uam.mx/))
* [BCI Competition II - Data set IIb](http://www.bbci.de/competition/ii/)
* [BCI Competition III - Data set II](http://www.bbci.de/competition/iii/)
* [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets)

## Requirements
* Python 3.7
* Tensorflow 1.14.0
* NumPy
* SciPy
* Pandas
* matplotlib
* scikit-learn
* cudatoolkit 10.0
* cudnn

You can create a [conda environment](https://www.anaconda.com/distribution/) with all the dependencies using the `environment.yml` file in this repository.

```
conda env create -n p300cnn -f environment.yml
```

## CNN Architectures
We evaluate the following state-of-the-art CNN architectures for within-subject and cross-subject P300 detection:

* CNN1 and CNN3 (as well as slight modifications of them)
  + Cecotti, H., & Graser, A. (2010). Convolutional neural networks for P300 detection with application to brain-computer interfaces. _IEEE transactions on pattern analysis and machine intelligence_, 33(3), 433-445. [[link](https://ieeexplore.ieee.org/document/5492691)]
* EEGNet ([@vlawhern's implementation](https://github.com/vlawhern/arl-eegmodels))
  + Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. _Journal of neural engineering_, 15(5), 056013. [[link](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c)] [[preprint](https://arxiv.org/abs/1611.08024)]
* ShallowConvNet and DeepConvNet ([@vlawhern's implementation](https://github.com/vlawhern/arl-eegmodels))
  + Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. _Human brain mapping_, 38(11), 5391-5420. [[link](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)] [[preprint](https://arxiv.org/abs/1703.05051)]
* OCLNN
  + Shan, H., Liu, Y., & Stefanov, T. P. (2018, July). A Simple Convolutional Neural Network for Accurate P300 Detection and Character Spelling in Brain Computer Interface. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, Stockholm, Sweden, 13-19 July 2018, 1604-1610. [[link](https://www.ijcai.org/Proceedings/2018/222)]
* BN<sup>3</sup>
  + Liu, M., Wu, W., Gu, Z., Yu, Z., Qi, F., & Li, Y. (2018). Deep learning based on batch normalization for P300 signal detection. _Neurocomputing_, 275, 288-297. [[link](https://www.sciencedirect.com/science/article/abs/pii/S0925231217314601)]
* CNN-R
  + Manor, R., & Geva, A. B. (2015) Convolutional neural network for multi-category rapid serial visual presentation BCI. _Frontiers in computational neuroscience_, 9, 146. [[link](https://www.frontiersin.org/articles/10.3389/fncom.2015.00146/full)]

We also propose and evaluate a simple CNN architecture (SepConv1D) (inspired by OCLNN) and a Fully-Connected Neural Network with a single hidden layer with two neurons (FCNN). The details of the architecture and the experimental results are reported in:
  + Alvarado-Gonzalez, A. M., Fuentes-Pineda, G., & Cervantes-Ojeda, J. (2021). A few filters are enough: convolutional neural network for P300 detection. _Neurocomputing_, 425, 37-52. [[link](https://www.sciencedirect.com/science/article/abs/pii/S0925231220317173)] [[preprint](https://arxiv.org/abs/1909.06970)]

## Results 
![alt text](figs/inference_time_auc.svg "Inference time vs AUC")

## Citation
```
@Article{p300cnnt_2021,
  author = {Montserrat Alvarado-González and Gibran Fuentes-Pineda and Jorge Cervantes-Ojeda},
  title = {A few filters are enough: Convolutional neural network for P300 detection},
  journal = {Neurocomputing},
  volume = {425},
  pages = {37--52},
  year = {2021},
}
```