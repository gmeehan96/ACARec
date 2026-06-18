# ACARec
This repository contains code and resources for the UMAP 2026 paper "[Leveraging Artist Catalogs for Cold-Start Music Recommendation](https://arxiv.org/abs/2604.07090)". The structure is based on [ColdRec](https://github.com/YuanchenBei/ColdRec), adapted for the artist-specific sampling necessary for ACARec's training.

The PyTorch code for [ACARec](https://github.com/gmeehan96/ACARec/blob/main/models/ACARec.py), as well as cold-start baselines, can be found in the [models](https://github.com/gmeehan96/ACARec/blob/main/models) folder.  Final hyperparameters for ACARec and baselines are in the [`main`](https://github.com/gmeehan96/ACARec/blob/main/main.py) and [`main_baseline`](https://github.com/gmeehan96/ACARec/blob/main/main_baseline.py) scripts respectively.

### Data preprocessing
In the [data](https://github.com/gmeehan96/ACARec/tree/main/data) folder, we include preprocessing scripts for our temporal splits of the [Music4All-Onion](https://zenodo.org/records/6609677) and [Yambda-50m](https://huggingface.co/datasets/yandex/yambda) datasets. These generate the necessary files for training the models. Some dataset files, such as the timestamp interaction data from Music4All-Onion, and the track metadata `id_information` file from the original Music4All, need to be downloaded first.

The exact splits and embeddings used in training our models can be found on [Google Drive](https://drive.google.com/drive/folders/19eQbDwkqRGn5HUO5ae-02pblVk1Seeyz?usp=sharing).
