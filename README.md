# Optimizing Data Usage via Differentiable Rewards
This is the code we used for the NMT experiments in our paper
>[Optimizing Data Usage via Differentiable Rewards](https://arxiv.org/pdf/1911.10088.pdf)

>Xinyi Wang, Hieu Pham, Paul Michel, Antonios Anastasopoulos, Jaime Carbonell, Graham Neubig


## Requirements

Python 3.7, PyTorch 1.1


## Data Processing

The data we use is [multilingual TED corpus](https://github.com/neulab/word-embeddings-for-nmt) by Qi et al.

We provide preprocessed version of the data, which you can get from [here](https://drive.google.com/file/d/1nTAOZ0_uI0sc7N77jkIVe-foHOXRz5In/view?usp=sharing):
If you are interested int the details of data processing, you can take a look at the script ``make-ted-multilingual.sh``.

## Training:

To run training with DDS+TCS on language bel, do
``bash scripts/dds_tcs.sh``
To run training with DDS on language bel, do
``bash scripts/dds.sh``


To run the baseline Uniform on language bel
``bash scripts/uniform.sh``


## Decoding:
``python scripts/trans.py OUTDIR``
Please replace OUTDIR with the directory of the saved model file.

## Note
We have a follow up work "[Balancing Training for Multilingual Neural Machine Translation](https://arxiv.org/abs/2004.06748)" with a similar algorithm implemented in fairseq. You could refer to the code linked in this paper for further reference.
