#!/bin/bash
# ============================================================================
# Precheck for a fresh BarcodeMAE clone before submitting the narval
# aux-weight-ablation jobs (bioscan5m_aux_weight_ablation_home.sh,
# its_aux_weight_ablation_home.sh) or its_export_tasks_home.sh.
#
# Runs directly on the login node -- everything here is cheap (file
# existence, python imports, `sbatch --test-only` dry-runs). No GPU needed,
# nothing is actually submitted to the queue.
#
# Run from the repo root:
#   bash slurm/final_scripts/narval_precheck.sh
# ============================================================================

REPO_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE_final/BarcodeMAE"
DATA_DIR_BIOSCAN="${REPO_DIR}/data/BIOSCAN-5M"
DATA_DIR_ITS="${REPO_DIR}/data/ITS-5M"
VENV_PATH="/scratch/$USER/BarcodeMAE_venv"
ACCOUNT="def-lila-ab"

N_PASS=0
N_WARN=0
N_FAIL=0

pass() { echo "  [PASS] $1"; N_PASS=$((N_PASS+1)); }
warn() { echo "  [WARN] $1"; N_WARN=$((N_WARN+1)); }
fail() { echo "  [FAIL] $1"; N_FAIL=$((N_FAIL+1)); }

echo "=========================================="
echo "narval precheck"
echo "User : $USER"
echo "Host : $(hostname)"
echo "Date : $(date)"
echo "=========================================="

# ── 1. Repo location and state ───────────────────────────────────────────────
echo ""
echo "-- Repo --"
if [ -d "$REPO_DIR" ]; then
    pass "Repo dir exists: $REPO_DIR"
    cd "$REPO_DIR" || exit 1
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        BRANCH="$(git branch --show-current 2>/dev/null)"
        LOCAL="$(git rev-parse HEAD 2>/dev/null)"
        REMOTE="$(git rev-parse @{u} 2>/dev/null)"
        pass "Git repo OK, branch '$BRANCH', HEAD $LOCAL"
        if [ -n "$REMOTE" ] && [ "$LOCAL" != "$REMOTE" ]; then
            warn "Local HEAD differs from upstream -- run 'git pull'"
        fi
        DIRTY="$(git status --porcelain)"
        [ -n "$DIRTY" ] && warn "Working tree has uncommitted changes"
        for f in slurm/final_scripts/bioscan5m_aux_weight_ablation_home.sh \
                 slurm/final_scripts/its_aux_weight_ablation_home.sh \
                 slurm/final_scripts/its_export_tasks_home.sh; do
            [ -f "$f" ] && pass "Found $f" || fail "Missing $f -- git pull needed"
        done
    else
        fail "$REPO_DIR is not a git repo"
    fi
else
    fail "Repo dir not found: $REPO_DIR"
fi

# ── 2. Venv + package smoke test ─────────────────────────────────────────────
echo ""
echo "-- Python environment --"
if [ -d "$VENV_PATH" ]; then
    pass "Venv exists: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    SMOKE_OUT=$(WANDB_MODE=disabled python -c "
import torch, torchtext, transformers, wandb
from pkg_resources import parse_version
from mycoai.data import Data
import barcodebert
print('OK')
" 2>&1)
    if echo "$SMOKE_OUT" | tail -1 | grep -q "^OK$"; then
        pass "All required packages import cleanly"
    else
        fail "Package import smoke test failed:"
        echo "$SMOKE_OUT" | sed 's/^/         /'
    fi
else
    fail "Venv not found at $VENV_PATH -- run slurm/setup_env.sh"
fi

# ── 3. SLURM account ──────────────────────────────────────────────────────────
echo ""
echo "-- SLURM account --"
if command -v sacctmgr >/dev/null 2>&1; then
    ASSOC=$(sacctmgr -n show assoc where user="$USER" account="$ACCOUNT" format=Account 2>/dev/null)
    if [ -n "$ASSOC" ]; then
        pass "Account '$ACCOUNT' is associated with user '$USER'"
    else
        fail "No association found for user '$USER' on account '$ACCOUNT' -- check the account name for this cluster"
    fi
else
    warn "sacctmgr not available -- cannot verify account association"
fi

# ── 4. Required data files ───────────────────────────────────────────────────
echo ""
echo "-- BIOSCAN-5M data ($DATA_DIR_BIOSCAN) --"
for f in pre_training.csv supervised_train.csv unseen.csv supervised_test.csv; do
    if [ -f "${DATA_DIR_BIOSCAN}/${f}" ]; then
        pass "$f present"
    else
        fail "$f missing"
    fi
done

echo ""
echo "-- ITS-5M data ($DATA_DIR_ITS) --"
for f in trainset.fasta trainset_valid.fasta test1.fasta test2.fasta; do
    if [ -f "${DATA_DIR_ITS}/${f}" ]; then
        pass "$f present"
    else
        fail "$f missing"
    fi
done

echo ""
echo "-- ITS-5M exported tasks ($DATA_DIR_ITS/tasks) --"
for f in test1_tasks.csv test2_tasks.csv; do
    if [ -f "${DATA_DIR_ITS}/tasks/${f}" ]; then
        pass "$f present"
    else
        warn "$f missing -- run: sbatch slurm/final_scripts/its_export_tasks_home.sh"
    fi
done

# ── 5. Writable checkpoint/wandb roots + rough quota check ───────────────────
echo ""
echo "-- Storage --"
for d in "${REPO_DIR}/main_checkpoints_final" "${REPO_DIR}/wandb_final" "${REPO_DIR}/results_final" "${REPO_DIR}/final_logs"; do
    if mkdir -p "$d" 2>/dev/null && touch "$d/.precheck_write_test" 2>/dev/null; then
        rm -f "$d/.precheck_write_test"
        pass "Writable: $d"
    else
        fail "Cannot write to: $d (check permissions / quota)"
    fi
done
if command -v diskusage_report >/dev/null 2>&1; then
    echo "  diskusage_report:"
    diskusage_report 2>/dev/null | sed 's/^/    /'
else
    warn "diskusage_report not available -- check quota manually (e.g. 'lfs quota' or cluster docs)"
fi

# ── 6. Dry-run the actual SLURM scripts (no submission) ──────────────────────
echo ""
echo "-- sbatch --test-only dry runs --"
if command -v sbatch >/dev/null 2>&1; then
    for script in slurm/final_scripts/bioscan5m_aux_weight_ablation_home.sh \
                  slurm/final_scripts/its_aux_weight_ablation_home.sh; do
        if [ -f "$script" ]; then
            OUT=$(sbatch --test-only --array=4 "$script" 2>&1)
            if echo "$OUT" | grep -qi "would start\|allocation"; then
                pass "$script: scheduler accepts it ($OUT)"
            else
                fail "$script: sbatch --test-only rejected it -- $OUT"
            fi
        fi
    done
    script="slurm/final_scripts/its_export_tasks_home.sh"
    if [ -f "$script" ]; then
        OUT=$(sbatch --test-only "$script" 2>&1)
        if echo "$OUT" | grep -qi "would start\|allocation"; then
            pass "$script: scheduler accepts it ($OUT)"
        else
            fail "$script: sbatch --test-only rejected it -- $OUT"
        fi
    fi
else
    warn "sbatch not found on this node -- skipping dry-run (are you on a login node?)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Summary: $N_PASS passed, $N_WARN warnings, $N_FAIL failed"
echo "=========================================="
[ "$N_FAIL" -gt 0 ] && exit 1
exit 0