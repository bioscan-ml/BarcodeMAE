# BarcodeMAE+

A PyTorch implementation of BarcodeMAE+, a model for enhancing DNA foundation models to address masking inefficiencies.

<p align="center">
  <img src="Figures/Arch_mae.png" alt="drawing" width="800"/>
</p>

#### Check out our paper (link coming soon)

#### Model checkpoint is available here: (link coming soon)

## Quick start

Use this jupyter notebook for quick start: (link coming soon)

## Setup

0. Clone this repository
1. Install the required libraries

```shell
pip install -r requirements.txt
pip install -e .
```

## Preparing the data

1. Download the metadata file and copy it into the data folder
2. Split the metadata file into smaller files according to the different partitions as presented in the [BIOSCAN-5M paper](https://arxiv.org/abs/2406.12723)

```shell
cd data/
python data_split.py BIOSCAN-5M_Dataset_metadata.tsv
```

## Reproducing the results

1. Download the checkpoint and copy it to the model_checkpoints directory
2. Run KNN evaluation

```shell
python barcodebert/knn_probing.py \
  --run-name knn_evaluation \
  --data-dir ./data/ \
  --pretrained-checkpoint "./model_checkpoints/best_pretraining.pt" \
  --log-wandb \
  --dataset BIOSCAN-5M
```

## Pretraining from scratch

1. Run pretraining

```shell
python barcodebert/pretraining.py \
  --dataset=BIOSCAN-5M \
  --k_mer=6 \
  --n_layers=6 \
  --n_heads=6 \
  --decoder-n-layers=6 \
  --decoder-n-heads=6 \
  --data_dir=data/ \
  --checkpoint=model_checkpoints/BIOSCAN-5M/6-6-6/model_checkpoint.pt
```

## Reproducing the baselines

`barcodebert/external_models.py` wraps off-the-shelf HuggingFace DNA foundation model checkpoints (DNABERT-2, DNABERT-S, Nucleotide Transformer, GENA-LM, Caduceus, HyenaDNA) so they can be evaluated with the same KNN pipeline used for BarcodeMAE+, via a separate dependency set (`requirements-external-baselines.txt` — see the file header for why this needs its own environment).

```shell
pip install -r requirements-external-baselines.txt

python barcodebert/knn_probing.py \
  --external-model-id <huggingface-model-id> \
  --data-dir ./data/ \
  --dataset BIOSCAN-5M
```

MycoAI (fungal ITS) and BarcodeMamba+ baselines use their own checkpoint formats and are evaluated with dedicated scripts:

```shell
# MycoAI-BERT / MycoAI-CNN on fungal ITS
python barcodebert/knn_its_mycoai.py --data-dir ./data/ --pretrained-checkpoint <path_to_mycoai_checkpoint>

# BarcodeMamba+ on BIOSCAN-5M
python barcodebert/knn_probing_barcodemamba.py --data-dir ./data/ --pretrained-checkpoint <path_to_barcodemamba_checkpoint>

# BarcodeMamba+ on fungal ITS
python barcodebert/knn_its_barcodemamba.py --data-dir ./data/ --pretrained-checkpoint <path_to_barcodemamba_checkpoint>
```

The original CNN and DNABERT baseline reproductions (used for the earlier BarcodeMAE arXiv paper) are kept under `scripts/CNN/` and `scripts/DNABERT/`.

Reference SLURM scripts for the full pretraining experiment grid (10 configurations: encoder-decoder vs. encoder-only, with/without CLS, and each auxiliary objective) are in `slurm/bioscan5m_final.sh` and `slurm/fungi_its_final.sh`. Update the SLURM account/paths at the top of each script for your own cluster before submitting.

## Citation

If you find BarcodeMAE+ useful in your research please consider citing:

```bibtex
@article{safari2025barcodemae,
  title={Enhancing DNA Foundation Models to Address Masking Inefficiencies},
  author={Monireh Safari
    and Pablo Millan Arias
    and Scott C. Lowe
    and Lila Kari
    and Angel X. Chang
    and Graham W. Taylor
  },
  journal={arXiv preprint arXiv:2502.18405},
  year={2025},
  eprint={2502.18405},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2502.18405},
}
```

<!-- BarcodeMAE+ (GigaScience, in preparation) citation to be added once available. -->