"""
Report validation-set masked-token-prediction accuracy for every aux-weight-ablation
checkpoint, alongside the (aux_task, weight) it corresponds to. Optionally also
includes the actual main-sweep checkpoint (weight=0.1, the baseline every
aux-weight-ablation script excludes since it already exists from
bioscan5m_final.sh / fungi_its_final.sh) for each aux_task, so the w=0.1 point
isn't missing from the comparison.

This is the SAME quantity pretraining.py already tracks and saves into every
checkpoint.pt during training (validation split, best epoch seen) -- no new
evaluation run is needed. This script just walks the checkpoint tree(s),
reads `max_accuracy`/`best_epoch`/`epoch` out of each checkpoint.pt, and
tabulates them by (aux_task, weight) so you can compare reconstruction quality
against the downstream KNN numbers already collected.

Usage:
    python barcodebert/report_aux_weight_val_accuracy.py \
        --ckpt-root /project/6045013/m4safari/BarcodeMAE/main_checkpoints_final/ablations/aux_weight \
        --main-ckpt-root /project/6045013/m4safari/BarcodeMAE/main_checkpoints_final \
        --dataset BIOSCAN-5M

    python barcodebert/report_aux_weight_val_accuracy.py \
        --ckpt-root /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE/main_checkpoints_final/ablations/aux_weight \
        --main-ckpt-root /home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/main_checkpoints_final \
        --dataset ITS-5M

--main-ckpt-root is optional -- omit it to report only the ablation sweep
(w != 0.1) as before. Note the main-sweep checkpoints (bioscan5m_final.sh /
fungi_its_final.sh) may live under a DIFFERENT base path than the aux-weight
ablation checkpoints even on the same cluster -- check where yours actually
landed (e.g. via `find . -iname 'final_k6_6L6H*cls_ce*' -o -iname
'final_its_k6_6L6H*cls_ce*'`) rather than assuming --ckpt-root's sibling.

Runs on CPU -- no GPU needed, safe to run on a login node or in a small
interactive session.
"""

import argparse
import os
import re
import sys

import torch

RUN_NAME_RE = re.compile(r"_cls_(?P<aux_task>ce|binary|triplet)_w(?P<weight>[0-9.]+)$")

# Main-sweep (weight=0.1 baseline) run-name convention, confirmed from
# slurm/bioscan5m_final.sh and slurm/fungi_its_final.sh.
MAIN_RUN_NAME = {
    "BIOSCAN-5M": "final_k6_6L6H_6DL6DH_maelm_cls_{aux_task}",
    "ITS-5M": "final_its_k6_6L6H_6DL6DH_maelm_cls_{aux_task}",
}


def find_checkpoints(ckpt_root, dataset):
    dataset_dir = os.path.join(ckpt_root, dataset)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: {dataset_dir} not found", file=sys.stderr)
        sys.exit(1)
    rows = []
    for run_name in sorted(os.listdir(dataset_dir)):
        run_dir = os.path.join(dataset_dir, run_name)
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            continue
        m = RUN_NAME_RE.search(run_name)
        if not m:
            print(f"WARNING: could not parse aux_task/weight from run name '{run_name}', skipping",
                  file=sys.stderr)
            continue
        rows.append((m.group("aux_task"), float(m.group("weight")), ckpt_path))
    return rows


def find_main_checkpoints(main_ckpt_root, dataset):
    """Locate the actual weight=0.1 main-sweep checkpoint for each aux_task."""
    dataset_dir = os.path.join(main_ckpt_root, dataset)
    rows = []
    for aux_task in ("ce", "binary", "triplet"):
        run_name = MAIN_RUN_NAME[dataset].format(aux_task=aux_task)
        ckpt_path = os.path.join(dataset_dir, run_name, "checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            print(f"WARNING: main-sweep checkpoint not found for aux_task={aux_task}: {ckpt_path}",
                  file=sys.stderr)
            continue
        rows.append((aux_task, 0.1, ckpt_path))
    return rows


def load_stats(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "epoch": checkpoint.get("epoch"),
        "max_accuracy": checkpoint.get("max_accuracy"),
        "best_epoch": checkpoint.get("best_epoch"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ckpt-root", required=True,
                    help="Path to .../main_checkpoints_final/ablations/aux_weight")
    p.add_argument("--main-ckpt-root", default=None,
                    help="Optional: path to .../main_checkpoints_final (the main-sweep "
                         "root, one level up from 'ablations'), to also include the "
                         "actual weight=0.1 baseline checkpoint for each aux_task.")
    p.add_argument("--dataset", required=True, choices=["BIOSCAN-5M", "ITS-5M"])
    args = p.parse_args()

    rows = find_checkpoints(args.ckpt_root, args.dataset)
    if args.main_ckpt_root:
        rows += find_main_checkpoints(args.main_ckpt_root, args.dataset)
    if not rows:
        print("No checkpoints found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for aux_task, weight, ckpt_path in rows:
        stats = load_stats(ckpt_path)
        results.append((aux_task, weight, stats))

    results.sort(key=lambda r: (r[0], r[1]))

    print(f"{'Aux task':<10} {'Weight':>15} {'Epoch':>6} {'Best epoch':>11} {'Val MLM acc (%)':>17}")
    print("-" * 63)
    for aux_task, weight, stats in results:
        epoch = stats["epoch"] if stats["epoch"] is not None else "--"
        best_epoch = stats["best_epoch"] if stats["best_epoch"] is not None else "--"
        max_acc = f"{stats['max_accuracy']:.4f}" if stats["max_accuracy"] is not None else "--"
        weight_label = f"{weight:.2f}" + (" (main)" if weight == 0.1 else "")
        print(f"{aux_task:<10} {weight_label:>15} {str(epoch):>6} {str(best_epoch):>11} {max_acc:>17}")


if __name__ == "__main__":
    main()