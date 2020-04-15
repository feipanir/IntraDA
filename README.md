# Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision

## Paper
[Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision](https://arxiv.org/abs/1811.12833)

Fei Pan, Inkyu Shin, Francois Rameau, Seokju Lee, In So Kweon
KAIST, Korea

IEEE Conference on Computer Vision and Pattern Recoginition (CVPR),2020 (**Oral**)

If you find this paper or code useful for your research, please cite our paper

## Abstract 
Convolutional neural network-based approaches have achieved remarkable progress in semantic segmentation. However, these approaches heavily rely on annotated data which are labor intensive. To cope with this limitation, automatically annotated data generated from graphic engines are used to train segmentation models. However, the models trained from synthetic data are difficult to transfer to real images. To tackle this issue, previous works have considered directly adapting models from the source data to the unlabeled target data (to reduce the inter-domain gap). Nonetheless, these techniques do not consider the large distribution gap among the target data itself (intra-domain gap). In this work, we propose a two-step self-supervised domain adaptation approach to minimize the inter-domain and intra-domain gap together. First, we conduct the inter-domain adaptation of the model; from this adaptation, we separate the target domain into an easy and hard split using an entropy-based ranking function. Finally, to decrease the intra-domain gap, we propose to employ a self-supervised adaptation technique from the easy to the hard split. Experimental results on numerous benchmark datasets highlight the effectiveness of our method against existing state-of-the-art approaches.

Our repo is modified from ADVENT.
### Pre-requsites (refere to ADVENT)
* Python 3.7
* Pytorch >= 0.4.1
* CUDA 9.0 or higher

### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/feipan664/IntraDA.git
$ cd IntraDA
```
1. Install ADVENT submodule and the dependices using pip:
```bash
$ pip install -e <root_dir/ADVENT>
```
With this, you can edit the ADVENT code on the fly and import function 
and classes of ADVENT in other project as well.

### Datasets
Similar to ADVENT, the datasets are put into ```<root_dir>/ADVENT/data```. Symlinks to hook the ADVENT codebase to the dataset is recommended. An alternative option is to explicitlly specify the parameters ```DATA_DIRECTORY_SOURCE``` and ```DATA_DIRECTORY_TARGET``` in YML configuration files.

* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
<root_dir>/data/GTA5/                               % GTA dataset root
<root_dir>/data/GTA5/images/                        % GTA images
<root_dir>/data/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/data/Cityscapes/                         % Cityscapes dataset root
<root_dir>/data/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/data/Cityscapes/leftImg8bit/val
<root_dir>/data/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/data/Cityscapes/gtFine/val
...
```

### Pre-trained models
Pre-trained models can be downloaded [here](https://github.com/valeoai/ADVENT/releases) and put in ```<root_dir>/ADVENT/pretrained_models```


## Running the code
For evaluation, execute:
```bash
$ cd <root_dir>/advent/scripts
$ python test.py --cfg ./configs/advent_pretrained.yml
$ python test.py --cfg ./configs/advent_cyclegan_pretrained.yml     % trained on cycleGAN translated images
```