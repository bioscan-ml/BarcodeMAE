import os
import json
import csv
import subprocess
import sys
try:
    import matplotlib
    matplotlib.use('Agg') # Force non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def parse_training_log(log_path):
    steps = []
    losses = []
    if not os.path.exists(log_path):
        return steps, losses
    
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[1] == 'train':
                try:
                    steps.append(int(parts[0]))
                    losses.append(float(parts[2]))
                except ValueError:
                    continue
    return steps, losses

def update_finetune_history(history_path, step, metrics):
    # metrics is a dict containing accuracy, etc.
    file_exists = os.path.exists(history_path)
    
    # Identify accuracy key
    acc_key = None
    for k in metrics.keys():
        if 'accuracy' in k or 'acc' in k:
            acc_key = k
            break
            
    if not acc_key and 'eval_accuracy' in metrics:
        acc_key = 'eval_accuracy'
        
    # We want to save at least step and accuracy
    # But let's save everything flat
    
    flat_metrics = {'step': step}
    flat_metrics.update(metrics)
    
    # We need to make sure we follow the header order if the file existed
    headers = []
    if file_exists:
         with open(history_path, 'r') as r:
            reader = csv.reader(r)
            try:
                headers = next(reader)
            except StopIteration:
                pass
    
    if not headers:
        headers = ['step'] + sorted(metrics.keys())
    else:
        # If new keys appeared, append them? Complicated for CSV.
        # Let's just stick to initial headers + any new ones appended if we were using pandas, but CSV reader is strict.
        # We'll just write what we can.
        new_keys = [k for k in metrics.keys() if k not in headers]
        headers.extend(new_keys)
        # Note: this might mess up previous rows if we were rewriting, but we are appending.
        # Actually appending with new columns is bad for CSV readers. 
        # Let's just stick to the headers we found or established.
    
    # Re-read to check if we need to rewrite header (not doing that for simplicity, assuming consistent metrics)
    
    with open(history_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        # Filter metrics to only include known headers
        row = {k: v for k, v in flat_metrics.items() if k in headers}
        writer.writerow(row)

def plot_curves(log_path, history_path, output_dir):
    if plt is None:
        print("Matplotlib not found, skipping plotting.")
        return

    train_steps, train_losses = parse_training_log(log_path)
    
    finetune_steps = []
    finetune_accs = []
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    finetune_steps.append(int(row['step']))
                    # Look for accuracy
                    acc = None
                    for k, v in row.items():
                        if 'accuracy' in k:
                           acc = float(v)
                           break
                    if acc is not None:
                        finetune_accs.append(acc)
                except ValueError:
                    continue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(train_steps, train_losses, label='Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True)
    
    if finetune_steps:
        ax2.plot(finetune_steps, finetune_accs, label='Downstream Accuracy', color='orange', marker='o')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Downstream Evaluation Accuracy')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_monitor.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Updated plots at {plot_path}")

def run_finetune(script_path, checkpoint_path, data_path, output_dir, step, model_type="bert"):
    finetune_out = os.path.join(output_dir, f"finetune_step_{step}")
    run_name = f"run_step_{step}"
    
    cmd = [
        sys.executable, script_path,
        "--model_name_or_path", checkpoint_path,
        "--model_type", model_type,
        "--data_path", data_path,
        "--output_dir", finetune_out,
        "--run_name", run_name,
        "--kmer", "-1",
        "--model_max_length", "128",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "16",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "3e-5",
        "--num_train_epochs", "3",
        "--fp16", # Flag
        "--save_steps", "200",
        "--evaluation_strategy", "steps",
        "--eval_steps", "200",
        "--warmup_steps", "50",
        "--logging_steps", "100000",
        "--overwrite_output_dir", "True",
        "--log_level", "info",
        "--find_unused_parameters", "False",
        "--save_model", "False", 
        "--eval_and_save_results", "True"
    ]
    
    # Add env var to disable wandb to avoid clutter
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    # Ensure finetune runs on a single GPU (the first one visible, or 0)
    # Since we are likely inside a distributed run where CUDA_VISIBLE_DEVICES might be set to "0,1,2,3"
    # we want to force this subprocess to only see one.
    # Actually, better to just let it see all but tell it to use only one via CUDA_VISIBLE_DEVICES override?
    # Or just rely on accelerator logic.
    # The traceback indicates the crash was NameError: 'sys', which is fixed above.
    # But regarding GPU usage:
    # If the parent process (train.py) holds onto GPU memory, this subprocess might OOM 
    # if it tries to allocate on the same GPU.
    # Ideally, we should set CUDA_VISIBLE_DEVICES to just one GPU, e.g. GPU 0.
    # But Rank 0 is running on GPU 0. It has the model loaded.
    # PyTorch usually doesn't release memory unless we close the process.
    # So running finetune.py (which creates a NEW model) on the SAME GPU might be tight on 64GB / 4 = 16GB?
    # No, you have 64GB total or per GPU? 
    # "mem=64G" in SLURM usually means System RAM.
    # "gres=gpu:h100:4" means 4 H100s. H100s have massive memory (80GB). 
    # So GPU memory overlap shouldn't be an issue.
    # However, to prevent DDP confusion in the subprocess, we should strictly set:
    env["CUDA_VISIBLE_DEVICES"] = "0" 
    # And ensure no distributed env vars leak that might confuse HF Trainer
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.pop("WORLD_SIZE", None)
    env.pop("MASTER_ADDR", None)
    env.pop("MASTER_PORT", None)
    
    print(f"Starting fine-tuning for step {step}...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fine-tuning failed: {e}")
        return {}
    
    # Parse results
    results_path = os.path.join(finetune_out, "results", run_name, "eval_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}
