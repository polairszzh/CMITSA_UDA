# Cross-Modal Interaction and Target-Specific Adaptation for Unsupervised Domain Adaptation

This repository contains the code for 'Cross-Modal Interaction and Target-Specific Adaptation for Unsupervised Domain Adaptation'(2026)

---
<div align="center">
  <img src="assets/framework.png" width="900px" />
</div>

---

## Installation

The model is built in Pytorch 2.11.0 and tested on Linux environment with NVIDIA RTX 5880 Ada Generation

For installing, follow these instructions:

```bash

conda create -n uda python=3.12
conda activate uda

```

Our code is built based on CLIP and Dassl, which can be installed with following commands.

```bash

# install CLIP

pip install git+https://github.com/openai/CLIP.git


# install Dassl

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git

cd dassl

pip install -r requirements.txt

pip install .

cd..

```

One can install other dependent tools via
```
pip install -r requirements.txt
```

## How to Download Datasets

The datasets used for UDA tasks can be downloaded via the following links.

VisDA17 (http://ai.bu.edu/visda-2017/#download)

Office-Home (https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)

Mini-DomainNet (http://ai.bu.edu/DomainNet/)

After downloading the datasets, please update the dataset paths in `scripts/{dataset}.sh` accordingly.

## How to Run the Code

We provide scripts for running UDA experiments on Office-Home, VisDA17, Mini-DomainNet datasets in the `scripts` folder.

For instance, to run a task on VisDA17:

```bash

cd scripts

sh VisDA17.sh

```

## Citation
If you find the code useful in your research, please consider citing:

    @InProceedings{cmista,
      title = {Cross-Modal Interaction and Target-Specific Adaptation for Unsupervised Domain Adaptation},
      author = {Zihong Zheng, Min Meng and Jigang Wu},
      year = {2026}
    }

## Acknowledgments

This project builds upon the invaluable contributions of following open-source projects:

1. DAPrompt (https://github.com/LeapLabTHU/DAPrompt)
2. CoOp (https://github.com/KaiyangZhou/CoOp)
3. DAMP (https://github.com/TL-UESTC/DAMP)

We express our sincere gratitude to the talented authors who have generously shared their source code with the public, enabling us to leverage their work in our own endeavor.