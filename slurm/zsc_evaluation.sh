#!/bin/bash
#SBATCH --job-name=zsc_eval
#SBATCH --account=def-lila-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsc_eval_%j.out
#SBATCH --error=logs/zsc_eval_%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "=========================================="

# Load modules
module load StdEnv/2023
module load cudacore/.12.6.3
module load python/3.11

echo "Loaded modules:"
module list

# Prevent Python from loading packages from ~/.local
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# Setup virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    virtualenv --no-download .venv
fi

source .venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies if needed
if [ ! -f ".venv/.h100_installed" ]; then
    echo "Installing dependencies for H100..."

    pip install --no-cache-dir --no-index --upgrade pip

    echo "Installing PyTorch 2.1.2 with CUDA 12.1 for H100..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    pip install --no-cache-dir -r requirements.txt
    pip install --no-cache-dir -e .

    touch .venv/.h100_installed
    echo "Dependencies installed."
else
    echo "Dependencies already installed for H100."
fi

# Verify GPU
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=========================================="
echo "Starting ZSC evaluation..."
echo "=========================================="

mkdir -p logs

CKPT_ROOT="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/model_checkpoints/BIOSCAN-5M"
DATA_DIR="/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/data/BIOSCAN-5M"
DATASET="BIOSCAN-5M"

# Format: "run_name|checkpoint_path|rep_type1,rep_type2,..."
MODELS=(
    "encoder_only_baseline|${CKPT_ROOT}/run_k6_6L_6H_0DL_0DH_jumbo0_transformer_encoder_0.01_genus_km8x4_64_0_128_0.00007_0.999_cls_n_m_0/best_pretraining.pt|all_tokens,tokens"
    "encoder_only_cls|${CKPT_ROOT}/run_k6_6L_6H_0DL_0DH_jumbo0_transformer_encoder_0.01_genus_km8x4_64_0_128_0.00007_0.999_clsreal_n_m_0/best_pretraining.pt|all_tokens,tokens,cls"
    "encoder_only_cls_registers|${CKPT_ROOT}/run_k6_6L_6H_0DL_0DH_jumbo0_transformer_encoder_0.01_genus_km8x4_64_0_128_0.00007_0.999_clsreal_n_m_6/best_pretraining.pt|all_tokens,tokens,cls,tokens_with_registers,all_with_registers"
    "encoder_only_jumbo6|${CKPT_ROOT}/run_k6_6L_6H_0DL_0DH_jumbo6_transformer_encoder_0.01_genus_km8x4_64_2_128_0.00007_0.999_n_m_0/best_pretraining.pt|all_tokens,tokens,jumbo_avg,jumbo"
    "encoder_decoder_baseline|${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_jumbo0_maelm_encoder_0.01_genus_km8x4_64_0_128_0.00007_0.999_n_m_normal/best_pretraining.pt|all_tokens,tokens"
    "encoder_decoder_cls|${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_jumbo0_maelm_encoder_0.01_genus_km8x4_64_0_128_0.00007_0.999_cls_n_m/best_pretraining.pt|all_tokens,tokens,cls"
    "encoder_decoder_cls_registers|${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_jumbo0_maelm_encoder_0.01_genus_km8x4_64_0_128_0.00007_0.999_cls_n_m_6/best_pretraining.pt|all_tokens,tokens,cls,tokens_with_registers,all_with_registers"
    "encoder_decoder_jumbo6|${CKPT_ROOT}/run_k6_6L_6H_6DL_6DH_jumbo6_maelm_encoder_0.01_genus_km8x4_64_2_128_0.00007_0.999_n_m_normal/best_pretraining.pt|all_tokens,tokens,jumbo_avg,jumbo"
)

OVERALL_EXIT=0

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r run_name ckpt_path rep_types <<< "$model_entry"

    echo ""
    echo "=========================================="
    echo "Model: ${run_name}"
    echo "Checkpoint: ${ckpt_path}"
    echo "=========================================="

    IFS=',' read -ra reps <<< "$rep_types"
    for rep_type in "${reps[@]}"; do
        echo ""
        echo "--- Representation type: ${rep_type} ---"

        python barcodebert/zsc_evaluation.py \
            --pretrained-checkpoint "${ckpt_path}" \
            --dataset "${DATASET}" \
            --data-dir "${DATA_DIR}" \
            --representation_type "${rep_type}" \
            --taxon genus \
            --n-neighbors 15 \
            --metric cosine \
            --run-name "zsc_${run_name}_${rep_type}" \
            --log-wandb

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "ERROR: zsc_evaluation failed for ${run_name} / ${rep_type} (exit ${EXIT_CODE})"
            OVERALL_EXIT=${EXIT_CODE}
        fi
    done
done

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Overall exit code: ${OVERALL_EXIT}"
echo "=========================================="

exit ${OVERALL_EXIT}