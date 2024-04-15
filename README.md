# Learning the meanings of function words from grounded language using a visual question answering model
 
This repository contains all the code for models, probes, and evaluations presented in the paper [*Learning the meanings of function words from grounded language using a visual question answering model*](./writeups/Function_words_and_VQA_manuscript.pdf) by Eva Portelance, Michael C. Frank, and Dan Jurafsky. 2023.

## Data

All the necessary preprocessed data -- the CLEVR dataset [(Johnson et al. 2017)](https://cs.stanford.edu/people/jcjohns/clevr/) partitions and semantic probes -- are available for download at [this OSF project repository](https://osf.io/4h5py/?view_only=3cf3d23c2f274f1487615ecfc9151d22).

More probe questions and partitions can also be created by using the scripts and notebooks provided in the folder [./dataset and probe creation/](./dataset%20and%20probe%20creation/).

## Model 

All pretrained models are available for download at [this OneDrive link](https://mcgill-my.sharepoint.com/:f:/g/personal/eva_portelance_mcgill_ca/EmmRTyeHKppHuViLHA9yi6kBvAka3Wta69UokYm0tYviyA?e=Y8NMW1).

The model code is available in the folder [./model and evaluations/mac-network](./model%20and%20evaluations/mac-network/). This folder contains a custom modified version of the MAC model by [Hudson & Manning (2018)](https://github.com/stanfordnlp/mac-network).

## Evaluation

All the evaluation and result analysis scripts are available in the folder [./model and evaluation/probe-evaluation-collection/](./model%20and%20evaluations/probe-evaluation-collection/). The results from the paper are also available for download at [this OSF project repository](https://osf.io/4h5py/?view_only=3cf3d23c2f274f1487615ecfc9151d22).


## To run code yourself
Use the following to set up a compatible environment. 

### Prerequisites
- Python 3.7 or higher
- pip package manager
- GPU (for faster training, optional but recommended)

### Installation
Clone the repository:

```bash
git clone https://github.com/evaportelance/vqa-function-word-learning.git
```
If you use anaconda, you can clone our environment using the conda-env.txt file:
```bash
cd vqa-function-word-learning
conda create --name myenv --file ./conda-env.txt
pip install requirements.txt
```

## Citation

Please cite the following paper:
```
@article{portelance2024learning,
  title={Learning the meanings of function words from grounded language using a visual question answering model},
  author={Portelance, Eva and Frank, Michael C. and Jurafsky, Dan},
  year={2024},
  journal={Cognitive Science}
}
```

Please also cite the papers for the MAC model and the CLEVR dataset:

```
@inproceedings{hudson2018compositional,
  title={Compositional Attention Networks for Machine Reasoning},
  author={Hudson, Drew A and Manning, Christopher D},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2018}
}
```
```
@inproceedings{johnson2017clevr,
  title={Clevr: A diagnostic dataset for compositional language and elementary visual reasoning},
  author={Johnson, Justin and Hariharan, Bharath and Van Der Maaten, Laurens and Fei-Fei, Li and Lawrence Zitnick, C and Girshick, Ross},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2901--2910},
  year={2017}
}
```
