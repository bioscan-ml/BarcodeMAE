# BarcodeMAE Implementation

This directory contains the core implementation of BarcodeMAE, including the MAE architecture, jumbo transformers, and taxonomy classification features.

## Directory Structure

```
barcodebert/
├── pretraining.py              # Main pretraining script
├── finetuning.py               # Supervised finetuning
├── knn_probing.py              # KNN evaluation
├── maelm_model.py              # MAE Language Model (encoder-decoder)
├── jumbo_transformer.py        # Jumbo BERT implementation
├── jumbo_transformer_with_taxonomy.py  # Transformer + taxonomy classifier
├── jumbo_taxonomy_classifier.py        # Binary taxonomy classifier
├── biological_masker.py        # Biologically-informed masking
├── datasets.py                 # Dataset classes (k-mer, BPE tokenization)
├── evaluation.py               # Evaluation utilities
├── io.py                       # Model loading/saving
├── utils.py                    # General utilities
├── test_maelm_shapes.py        # Unit tests for MAELM
├── test_transformer_taxonomy.py # Unit tests for transformer
└── [Documentation]
    ├── README.md               # This file
    ├── TAXONOMY_CLASSIFICATION_SUMMARY.md  # Quick reference
    ├── JUMBO_SOURCE_GUIDE.md   # MAELM jumbo sources guide
    ├── TRANSFORMER_TAXONOMY_GUIDE.md       # Transformer guide
    └── JUMBO_SOURCE_SUMMARY.md # Implementation summary
```

## Key Components

### 1. Model Architectures

#### MAELMModel (`maelm_model.py`)
The encoder-decoder masked autoencoder architecture:
- **Encoder**: Can be standard BERT or Jumbo BERT
- **Decoder**: Standard BERT with token classification head
- **Jumbo Source Options**:
  - `encoder`: Use jumbo tokens directly from encoder
  - `decoder`: Use jumbo tokens after decoder processing

```python
from barcodebert.maelm_model import MAELMModel

model = MAELMModel(
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    jumbo=True,
    jumbo_multiplier=6,
    share_jumbo_layers=False,
    enable_genus_classification=True,
    jumbo_source="decoder"  # or "encoder"
)
```

#### JumboTransformerWithTaxonomy (`jumbo_transformer_with_taxonomy.py`)
Single-encoder transformer with jumbo tokens and taxonomy classification:
- Simpler than MAELM (no decoder)
- Faster training
- Standard masked language modeling

```python
from barcodebert.jumbo_transformer_with_taxonomy import create_jumbo_transformer_with_taxonomy

model = create_jumbo_transformer_with_taxonomy(
    bert_config=config,
    jumbo_multiplier=6,
    enable_taxonomy_classification=True
)
```

### 2. Jumbo CLS Tokens (`jumbo_transformer.py`)

Implementation of Jumbo CLS tokens:
- **J** learnable CLS tokens prepended to input
- Processed through wide MLP (J × D flattened dimension)
- Aggregates global sequence information
- Used for downstream tasks (taxonomy classification)

Key class: `JumboTokenHandler`
- Flattens J tokens: `(B, J, D) → (B, J×D)`
- Applies wide MLP with residual connection
- Reshapes back: `(B, J×D) → (B, J, D)`

### 3. Taxonomy Classification (`jumbo_taxonomy_classifier.py`)

Binary pairwise taxonomy classifier:
- Takes two jumbo representations
- Predicts if they share the same taxonomic label
- Used during pretraining as auxiliary task

```python
from barcodebert.jumbo_taxonomy_classifier import (
    JumboTaxonomyClassifier,
    compute_taxonomy_classification_loss
)

# Create classifier
classifier = JumboTaxonomyClassifier(jumbo_dim=2304)  # 6 × 384

# Compute loss during training
taxonomy_loss, accuracy, num_pairs, num_same, num_diff = \
    compute_taxonomy_classification_loss(
        jumbo_tokens,
        genus_labels,
        classifier,
        same_ratio=0.5
    )
```

### 4. Biological Masking (`biological_masker.py`)

Temperature-based biologically-informed masking:
- Uses precomputed substitution matrices
- Masks tokens with biologically similar alternatives
- Temperature controls substitution difficulty

```python
from biological_masker import TemperatureCompatibleBiologicalMasker

masker = TemperatureCompatibleBiologicalMasker(
    vocab=vocab,
    k_mer=6,
    temperature=1.0
)
```

### 5. Datasets (`datasets.py`)

Dataset classes with tokenization:
- **K-mer tokenization**: Sliding window of k nucleotides
- **BPE tokenization**: Byte-pair encoding

```python
from barcodebert.datasets import DNADataset

dataset = DNADataset(
    metadata_path="data/train_metadata.tsv",
    tokenizer_type="kmer",
    k_mer=6,
    stride=1,
    max_len=660
)
```

## Usage Examples

### Standard Pretraining (No Taxonomy)

```bash
# MAELM architecture
python pretraining.py \
  --arch maelm \
  --dataset BIOSCAN-5M \
  --data_dir data/

# Transformer architecture
python pretraining.py \
  --arch transformer \
  --dataset BIOSCAN-5M \
  --data_dir data/
```

### With Jumbo Tokens (No Taxonomy)

```bash
# MAELM with jumbo encoder
python pretraining.py \
  --arch maelm \
  --jumbo \
  --jumbo_multiplier 6

# Transformer with jumbo
python pretraining.py \
  --arch transformer \
  --jumbo \
  --jumbo_multiplier 6
```

### With Taxonomy Classification

```bash
# MAELM + encoder source
python pretraining.py \
  --arch maelm \
  --jumbo \
  --enable-taxonomy-classification \
  --jumbo-source encoder \
  --taxonomy-level genus

# MAELM + decoder source
python pretraining.py \
  --arch maelm \
  --jumbo \
  --enable-taxonomy-classification \
  --jumbo-source decoder \
  --taxonomy-level genus

# Transformer
python pretraining.py \
  --arch transformer \
  --jumbo \
  --enable-taxonomy-classification \
  --taxonomy-level genus
```

## Testing

### Run All Tests

```bash
# Test MAELM shapes and functionality
python -m barcodebert.test_maelm_shapes

# Test transformer with taxonomy
python -m barcodebert.test_transformer_taxonomy
```

### Expected Test Output

```
✓ All tests passed!

Key findings:
1. Standard MAE (no jumbo): logits (B, seq_len, vocab_size)
2. Jumbo MAE with encoder source: jumbo_tokens (B, J, D)
3. Jumbo MAE with decoder source: jumbo_tokens (B, J, D) [different values]
4. Taxonomy classification loss computation works
5. Gradients flow correctly
```

## Training Pipeline

### 1. Data Preparation
```bash
cd data/
python data_split.py BIOSCAN-5M_Dataset_metadata.tsv
```

### 2. Pretraining
```bash
python barcodebert/pretraining.py \
  --arch maelm \
  --jumbo \
  --enable-taxonomy-classification \
  --jumbo-source decoder \
  --taxonomy-level genus \
  --checkpoint model_checkpoints/pretrained.pt
```

### 3. Evaluation
```bash
python barcodebert/knn_probing.py \
  --pretrained-checkpoint model_checkpoints/pretrained.pt \
  --dataset BIOSCAN-5M \
  --data-dir data/
```

### 4. Finetuning (Optional)
```bash
python barcodebert/finetuning.py \
  --pretrained-checkpoint model_checkpoints/pretrained.pt \
  --dataset BIOSCAN-5M \
  --data_dir data/ \
  --checkpoint model_checkpoints/finetuned.pt
```

## Key Configuration Parameters

### Architecture
- `--arch`: Model architecture (`maelm` or `transformer`)
- `--n_layers`: Number of encoder layers (default: 6)
- `--n_heads`: Number of attention heads (default: 6)
- `--embed_dim`: Hidden dimension (default: 384)

### Decoder (MAELM only)
- `--decoder-n-layers`: Number of decoder layers
- `--decoder-n-heads`: Number of decoder attention heads
- `--decoder-embed-dim`: Decoder hidden dimension

### Jumbo Configuration
- `--jumbo`: Enable jumbo CLS tokens
- `--jumbo_multiplier`: Number of jumbo tokens (default: 6)
- `--share_jumbo_layers`: Share jumbo MLP across layers

### Taxonomy Classification
- `--enable-taxonomy-classification`: Enable taxonomy head
- `--jumbo-source`: Source for MAELM (`encoder` or `decoder`)
- `--taxonomy-level`: Level (`phylum`, `class`, `order`, `family`, `genus`, `species`)
- `--taxonomy-loss-weight`: Weight for taxonomy loss (default: 0.1)

### Masking
- `--mask_ratio`: Fraction of tokens to mask (default: 0.4)
- `--biological_masking`: Use biologically-informed masking
- `--temperature`: Temperature for biological masking

### Training
- `--batch-size-per-gpu`: Batch size per GPU
- `--learning_rate`: Learning rate (default: 1e-4)
- `--epochs`: Number of training epochs
- `--log-wandb`: Log to Weights & Biases

## Documentation

### Quick Reference
- **[TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md)**: Overview of all options

### Detailed Guides
- **[JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md)**: Comprehensive guide for MAELM jumbo sources
- **[TRANSFORMER_TAXONOMY_GUIDE.md](TRANSFORMER_TAXONOMY_GUIDE.md)**: Guide for transformer architecture
- **[JUMBO_SOURCE_SUMMARY.md](JUMBO_SOURCE_SUMMARY.md)**: Implementation summary

## Model Outputs

### MAELM
```python
outputs = model(input_ids, attention_mask, mask_positions)

# Outputs contain:
outputs.logits          # (B, seq_len, vocab_size) - reconstruction logits
outputs.jumbo_tokens    # (B, J, D) - jumbo CLS tokens for classification
```

### Transformer
```python
outputs = model(input_ids, attention_mask)

# Outputs contain:
outputs.logits                  # (B, seq_len, vocab_size)
outputs.hidden_states           # (B, seq_len, D)
outputs.jumbo_tokens            # (B, J, D)
outputs.jumbo_representation    # (B, J×D) - flattened for classifier
```

## Loading Pretrained Models

```python
from barcodebert.io import load_pretrained_model

model, checkpoint = load_pretrained_model(
    checkpoint_path="model_checkpoints/pretrained.pt",
    device="cuda"
)

# Access configuration
config = checkpoint["config"]
bert_config = checkpoint["bert_config"]
```

## Advanced Features

### Distributed Training
The codebase supports multi-GPU distributed training via PyTorch DDP:

```bash
# Will automatically detect SLURM environment
python pretraining.py \
  --distributed \
  --batch-size-per-gpu 32
```

### Weights & Biases Logging
```bash
python pretraining.py \
  --log-wandb \
  --run-name my_experiment \
  --wandb-project BarcodeMAE
```

### Custom Masking Strategies
```bash
# Biological masking with temperature
python pretraining.py \
  --biological_masking \
  --temperature 1.0 \
  --temperature_schedule cosine
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'barcodebert'`
**Solution**: Install the package: `pip install -e .` from the BarcodeMAE directory

**Issue**: Taxonomy classification not working
**Solution**: Ensure `--jumbo` is enabled along with `--enable-taxonomy-classification`

**Issue**: CUDA out of memory
**Solution**: Reduce `--batch-size-per-gpu` or `--max_len`

**Issue**: Slow training
**Solution**: Use `--share_jumbo_layers` to reduce parameters, or use `--arch transformer` for simpler model

## Performance Tips

1. **Use shared jumbo layers** (`--share_jumbo_layers`) to reduce memory and speed up training
2. **Adjust batch size** based on GPU memory
3. **Use transformer architecture** for fastest training
4. **Enable mixed precision** training (automatically enabled with CUDA)
5. **Use distributed training** for multiple GPUs

## Contributing

When adding new features:
1. Add unit tests in `test_*.py` files
2. Update relevant documentation files
3. Follow existing code style (Black formatter, 120 char line length)
4. Add docstrings to new functions/classes

## Support

For questions or issues:
1. Check the documentation files in this directory
2. Run the unit tests to verify your setup
3. Open an issue on GitHub with error details and configuration