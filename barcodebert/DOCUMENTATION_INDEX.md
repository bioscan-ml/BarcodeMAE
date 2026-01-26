# BarcodeMAE Documentation Index

Complete guide to all BarcodeMAE documentation files.

## Getting Started

### New Users Start Here
1. **[Main README](../README.md)** - Project overview, setup, and quick start
2. **[BarcodeMAE Paper](https://arxiv.org/pdf/2502.18405)** - Full technical details
3. **[Quick Start Notebook](https://colab.research.google.com/drive/17fXgva89PzC29cgegjqsuKPZ6KGXpguc?usp=sharing)** - Hands-on tutorial

### Implementation Details
4. **[barcodebert/README.md](README.md)** - Code structure, API reference, implementation guide

## Taxonomy Classification Documentation

### Quick Reference
- **[TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md)**
  - Overview of all taxonomy classification options
  - Quick command reference
  - When to use which architecture
  - **Start here** for taxonomy classification

### Detailed Guides

#### MAELM Architecture (Encoder-Decoder)
- **[JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md)**
  - Comprehensive guide for encoder vs decoder jumbo sources
  - How each option works
  - Performance considerations
  - Use cases and recommendations
  - Troubleshooting

- **[JUMBO_SOURCE_SUMMARY.md](JUMBO_SOURCE_SUMMARY.md)**
  - Implementation summary
  - Architecture flow diagrams
  - Files modified
  - Validation rules

#### Transformer Architecture (Single Encoder)
- **[TRANSFORMER_TAXONOMY_GUIDE.md](TRANSFORMER_TAXONOMY_GUIDE.md)**
  - Complete guide for transformer with taxonomy classification
  - Comparison with MAELM
  - Usage examples
  - Performance tips
  - When to use transformer vs MAELM

## Documentation by Topic

### Architecture Guides

| Topic | File | Description |
|-------|------|-------------|
| **MAELM Overview** | [JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md) | Encoder-decoder MAE with jumbo tokens |
| **Transformer Overview** | [TRANSFORMER_TAXONOMY_GUIDE.md](TRANSFORMER_TAXONOMY_GUIDE.md) | Single-encoder with jumbo tokens |
| **Architecture Comparison** | [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md) | Compare all options |

### Feature Guides

| Feature | File | Description |
|---------|------|-------------|
| **Jumbo CLS Tokens** | [barcodebert/README.md](README.md) | Implementation details |
| **Taxonomy Classification** | [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md) | Binary pairwise classifier |
| **Jumbo Sources** | [JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md) | Encoder vs decoder sources |
| **Biological Masking** | [barcodebert/README.md](README.md) | Temperature-based masking |

### Code Reference

| Component | File | Description |
|-----------|------|-------------|
| **MAELMModel** | `maelm_model.py` | Encoder-decoder implementation |
| **JumboTransformer** | `jumbo_transformer.py` | Jumbo BERT implementation |
| **JumboTransformerWithTaxonomy** | `jumbo_transformer_with_taxonomy.py` | Transformer + taxonomy |
| **JumboTaxonomyClassifier** | `jumbo_taxonomy_classifier.py` | Binary classifier |
| **Pretraining** | `pretraining.py` | Main training script |
| **KNN Probing** | `knn_probing.py` | Evaluation script |

## Common Use Cases

### 1. Standard Pretraining (No Taxonomy)
**Files to read:**
- [Main README](../README.md) - Setup and basic usage
- [barcodebert/README.md](README.md) - Configuration parameters

**Command:**
```bash
python barcodebert/pretraining.py \
  --arch maelm \
  --dataset BIOSCAN-5M \
  --data_dir data/
```

### 2. Pretraining with Taxonomy Classification
**Files to read:**
- [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md) - Quick reference
- [JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md) - Detailed MAELM guide
- [TRANSFORMER_TAXONOMY_GUIDE.md](TRANSFORMER_TAXONOMY_GUIDE.md) - Transformer guide

**Commands:**
```bash
# MAELM encoder source
python barcodebert/pretraining.py \
  --arch maelm --jumbo \
  --enable-taxonomy-classification \
  --jumbo-source encoder

# MAELM decoder source
python barcodebert/pretraining.py \
  --arch maelm --jumbo \
  --enable-taxonomy-classification \
  --jumbo-source decoder

# Transformer
python barcodebert/pretraining.py \
  --arch transformer --jumbo \
  --enable-taxonomy-classification
```

### 3. Comparing Different Architectures
**Files to read:**
- [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md) - Architecture comparison
- [JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md) - MAELM options
- [TRANSFORMER_TAXONOMY_GUIDE.md](TRANSFORMER_TAXONOMY_GUIDE.md) - Transformer option

**Approach:**
Train three models and compare KNN results:
1. MAELM encoder source
2. MAELM decoder source
3. Transformer

### 4. Custom Taxonomy Levels
**Files to read:**
- [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md) - Parameter reference
- [barcodebert/README.md](README.md) - Detailed parameters

**Commands:**
```bash
# Family level
python pretraining.py --taxonomy-level family

# Species level
python pretraining.py --taxonomy-level species
```

### 5. Testing and Validation
**Files to read:**
- [barcodebert/README.md](README.md) - Testing section

**Commands:**
```bash
# Test MAELM
python -m barcodebert.test_maelm_shapes

# Test transformer
python -m barcodebert.test_transformer_taxonomy
```

## Quick Command Reference

### MAELM Commands

```bash
# Standard MAELM
python pretraining.py --arch maelm

# MAELM with jumbo (no taxonomy)
python pretraining.py --arch maelm --jumbo

# MAELM + encoder source taxonomy
python pretraining.py --arch maelm --jumbo \
  --enable-taxonomy-classification --jumbo-source encoder

# MAELM + decoder source taxonomy
python pretraining.py --arch maelm --jumbo \
  --enable-taxonomy-classification --jumbo-source decoder
```

### Transformer Commands

```bash
# Standard transformer
python pretraining.py --arch transformer

# Transformer with jumbo (no taxonomy)
python pretraining.py --arch transformer --jumbo

# Transformer + taxonomy
python pretraining.py --arch transformer --jumbo \
  --enable-taxonomy-classification
```

### Evaluation Commands

```bash
# KNN probing
python barcodebert/knn_probing.py \
  --pretrained-checkpoint model_checkpoints/pretrained.pt \
  --dataset BIOSCAN-5M

# Finetuning
python barcodebert/finetuning.py \
  --pretrained-checkpoint model_checkpoints/pretrained.pt
```

## File Organization

```
BarcodeMAE/
├── README.md                           # Main project README
├── requirements.txt                    # Python dependencies
└── barcodebert/
    ├── README.md                       # Implementation guide
    ├── DOCUMENTATION_INDEX.md          # This file
    ├── TAXONOMY_CLASSIFICATION_SUMMARY.md  # Quick reference
    ├── JUMBO_SOURCE_GUIDE.md           # MAELM detailed guide
    ├── JUMBO_SOURCE_SUMMARY.md         # Implementation summary
    ├── TRANSFORMER_TAXONOMY_GUIDE.md   # Transformer guide
    ├── pretraining.py                  # Main training script
    ├── maelm_model.py                  # MAELM implementation
    ├── jumbo_transformer.py            # Jumbo BERT
    ├── jumbo_transformer_with_taxonomy.py  # Transformer + taxonomy
    ├── jumbo_taxonomy_classifier.py    # Taxonomy classifier
    ├── test_maelm_shapes.py            # MAELM tests
    └── test_transformer_taxonomy.py    # Transformer tests
```

## Finding Information

### By Question

**"How do I enable taxonomy classification?"**
→ [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md)

**"What's the difference between encoder and decoder sources?"**
→ [JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md) - Section "Key Differences"

**"Should I use MAELM or Transformer?"**
→ [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md) - Section "When to Use Which"

**"How do jumbo tokens work?"**
→ [barcodebert/README.md](README.md) - Section "Jumbo CLS Tokens"

**"What parameters can I configure?"**
→ [barcodebert/README.md](README.md) - Section "Key Configuration Parameters"

**"How do I test my installation?"**
→ [barcodebert/README.md](README.md) - Section "Testing"

**"How is taxonomy loss computed?"**
→ [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md) - Section "How It Works"

### By Experience Level

**Beginner (First Time User)**
1. [Main README](../README.md)
2. [Quick Start Notebook](https://colab.research.google.com/drive/17fXgva89PzC29cgegjqsuKPZ6KGXpguc?usp=sharing)
3. [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md)

**Intermediate (Want to Use Taxonomy Classification)**
1. [TAXONOMY_CLASSIFICATION_SUMMARY.md](TAXONOMY_CLASSIFICATION_SUMMARY.md)
2. [JUMBO_SOURCE_GUIDE.md](JUMBO_SOURCE_GUIDE.md) OR [TRANSFORMER_TAXONOMY_GUIDE.md](TRANSFORMER_TAXONOMY_GUIDE.md)
3. [barcodebert/README.md](README.md) - Advanced Features

**Advanced (Want to Modify/Extend)**
1. [barcodebert/README.md](README.md) - Full implementation details
2. Source code files (`.py`)
3. Test files (`test_*.py`)

## External Resources

- **Paper**: [Enhancing DNA Foundation Models](https://arxiv.org/pdf/2502.18405)
- **Colab Notebook**: [Quick Start](https://colab.research.google.com/drive/17fXgva89PzC29cgegjqsuKPZ6KGXpguc?usp=sharing)
- **Model Checkpoint**: [Google Drive](https://drive.google.com/file/d/18TqKC_gLYYDZEFfkMBRvWTHTT8Vb74Wv/view?usp=drive_link)
- **BIOSCAN-5M Dataset**: [Paper](https://arxiv.org/abs/2406.12723)

## Support and Issues

1. **Check documentation** in this index
2. **Run tests** to verify installation:
   ```bash
   python -m barcodebert.test_maelm_shapes
   python -m barcodebert.test_transformer_taxonomy
   ```
3. **Review examples** in the documentation files
4. **Open an issue** on GitHub with:
   - Error message
   - Command used
   - Configuration parameters
   - Output of test scripts

## Contributing to Documentation

When adding new features:
1. Update relevant documentation files
2. Add entry to this index
3. Update main README if user-facing
4. Add examples to appropriate guide
5. Include unit tests

## Documentation Standards

All documentation files follow these standards:
- **Markdown format** for compatibility
- **Code examples** with full commands
- **Table of contents** for long documents
- **Cross-references** between related docs
- **Quick start** sections at the beginning
- **Troubleshooting** sections when applicable

---

**Last Updated**: 2026-01-26

**Documentation Version**: Includes taxonomy classification features (MAELM encoder/decoder sources, transformer architecture support)
