r"""
Print the exact parameter count for every model in Tables 5/6 (baseline comparison
tables), plus our own BarcodeMAE+ encoder, so the manuscript's "# Parameters" column
can be filled in with real numbers instead of guessed ones.

Usage (on the cluster, from the BarcodeMAE/ directory, with the usual env activated):

    python barcodebert/count_baseline_params.py

Optionally restrict to a subset:

    python barcodebert/count_baseline_params.py --skip-mycoai
    python barcodebert/count_baseline_params.py --skip-hf

MycoAI checkpoint paths default to /scratch/$USER/mycoai_models/{MycoAI-BERT,MycoAI-CNN}.pt
(where earlier debugging in this session moved them); override with --mycoai-bert-ckpt /
--mycoai-cnn-ckpt if yours live elsewhere. Our own checkpoints default to the
main_checkpoints_final/ paths used throughout the paper; override with --bioscan-ckpt /
--its-ckpt if needed.
"""

import argparse
import os

import torch

# Same HF model IDs used by the SLURM baseline scripts
# (slurm/final_scripts/bioscan5m_external_baselines.sh, bioscan5m_external_baselines_modern.sh,
# its5m_external_baselines.sh, its5m_external_baselines_modern.sh).
HF_MODELS = [
    ("DNABERT-2", "zhihan1996/DNABERT-2-117M"),
    ("DNABERT-S", "zhihan1996/DNABERT-S"),
    ("Nucleotide Transformer", "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"),
    ("BarcodeBERT", "bioscan-ml/BarcodeBERT"),
    ("HyenaDNA-tiny", "LongSafari/hyenadna-tiny-1k-seqlen-hf"),
    ("Caduceus-PS-1k", "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"),
    ("GENA-LM", "AIRI-Institute/moderngena-base"),
]


def count_hf_models():
    from transformers import AutoModel

    print("\n=== HuggingFace baseline models ===")
    results = {}
    for name, model_id in HF_MODELS:
        try:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"{name:28s} ({model_id}): {n_params:,} parameters")
            results[name] = n_params
            del model
        except Exception as e:
            print(f"{name:28s} ({model_id}): FAILED -- {e}")
    return results


def count_mycoai(bert_ckpt, cnn_ckpt):
    from mycoai import utils as mycoai_utils

    print("\n=== MycoAI baseline models ===")
    results = {}
    for name, ckpt_path in [("MycoAI-BERT", bert_ckpt), ("MycoAI-CNN", cnn_ckpt)]:
        if not os.path.exists(ckpt_path):
            print(f"{name:28s}: checkpoint not found at {ckpt_path}, skipping")
            continue
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:28s} ({ckpt_path}): {n_params:,} parameters")
        results[name] = n_params
        del model
    return results


def count_ours(bioscan_ckpt, its_ckpt):
    # NOTE: for arch=="maelm", load_pretrained_model() (barcodebert/io.py) builds
    # `model = BertModel(bert_config)` and loads ONLY the "encoder."-prefixed
    # weights from the checkpoint -- it never reconstructs the decoder at all.
    # So the returned `model` object already *is* the full encoder (embeddings +
    # transformer layers + pooler); it has no `.decoder` attribute. Do not call
    # `model.encoder` -- BertModel has its own internal submodule also named
    # `.encoder` (just the transformer-layer stack), which would silently
    # undercount by excluding the embedding table and pooler.
    from barcodebert.io import load_pretrained_model

    print("\n=== BarcodeMAE+ (ours) ===")
    results = {}
    for name, ckpt_path in [("BarcodeMAE+ (BIOSCAN-5M)", bioscan_ckpt), ("BarcodeMAE+ (UNITE+INSD)", its_ckpt)]:
        if not os.path.exists(ckpt_path):
            print(f"{name:28s}: checkpoint not found at {ckpt_path}, skipping")
            continue
        model, ckpt = load_pretrained_model(ckpt_path, device="cpu")
        encoder_params = sum(p.numel() for p in model.parameters())
        print(f"{name:28s}: {encoder_params:,} encoder parameters (the only part used at inference; "
              f"the decoder is discarded after pretraining and is not loaded here)")
        results[name] = encoder_params
        del model
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-mycoai", action="store_true")
    parser.add_argument("--skip-ours", action="store_true")
    parser.add_argument(
        "--mycoai-bert-ckpt",
        default=os.path.join("/scratch", os.environ.get("USER", ""), "mycoai_models", "MycoAI-BERT.pt"),
    )
    parser.add_argument(
        "--mycoai-cnn-ckpt",
        default=os.path.join("/scratch", os.environ.get("USER", ""), "mycoai_models", "MycoAI-CNN.pt"),
    )
    parser.add_argument(
        "--bioscan-ckpt",
        default="main_checkpoints_final/BIOSCAN-5M/final_k6_6L6H_6DL6DH_maelm_cls_ce/checkpoint_encoder.pt",
    )
    parser.add_argument(
        "--its-ckpt",
        default="main_checkpoints_final/UNITE-INSD/final_its_k6_6L6H_6DL6DH_maelm_cls_binary/checkpoint_encoder.pt",
    )
    args = parser.parse_args()

    all_results = {}
    if not args.skip_hf:
        all_results.update(count_hf_models())
    if not args.skip_mycoai:
        all_results.update(count_mycoai(args.mycoai_bert_ckpt, args.mycoai_cnn_ckpt))
    if not args.skip_ours:
        all_results.update(count_ours(args.bioscan_ckpt, args.its_ckpt))

    print("\n=== Summary (copy into manuscript) ===")
    for name, n in all_results.items():
        print(f"{name}: {n:,}")


if __name__ == "__main__":
    main()