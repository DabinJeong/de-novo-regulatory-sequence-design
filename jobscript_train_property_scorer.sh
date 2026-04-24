#BSUB -J train-sequence_property_scorer
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o train-sequence_property_scorer.%J.out
#BSUB -e train-sequence_property_scorer.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 16
#BSUB -M 40GB
#BSUB -R "select[mem>40GB] rusage[mem=40GB] span[hosts=1]"

set -e

module load python-3.11.6

# nvidia-smi
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="/nfs/team361/dj16/projects/sequence_generation/"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sequence_generation/outputs"

ENV_PATH="/nfs/team361/dj16/pypoetry/virtualenvs/sequence-generation-7Ds7Y9Ey-py3.12/bin/activate"

[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# === Training ===
# # Without DDP
python -u -m scripts.main --train --config ${run_dir}/configs/enhancer_gosai_masked_separator.yaml --out_dir ${output_dir}/sequence_generation_${LSB_JOBID}

# With DDP
# torchrun --standalone --nproc_per_node 2 -m scripts.main --train --config ${run_dir}/configs/enhancer_gosai.yaml --out_dir ${output_dir}/sequence_generation_${LSB_JOBID}

# === Train ensemble for guided generation === 
if [ ! -f ./runs/ensemble/ensemble_best.ckpt ];then
    python -m scripts.main_guided --config configs/enhancer_gosai_guided.yaml --out_dir ./runs/ensemble --train_ensemble
fi

# === Analyze seq contents of invariant/env subsequeces ===
python -m scripts.analyze_invariant_separator --config configs/enhancer_gosai_masked_separator.yaml --ckpt runs/masked_separator/masked_separator_best.ckpt --out_dir runs/masked_separator/analysis --num_samples 4000 --split val

# === Visualize mask ===
python -m scripts.visualize_mask_structure --config configs/enhancer_gosai_masked_separator.yaml --ckpt   runs/masked_separator/masked_separator_best.ckpt --out_dir runs/masked_separator/analysis/mask_structure --num_samples 1000 --split val

# === JASPAR motif hits: invariant vs env region (full-seq scan, density-normalised) ===
python -m scripts.analyze_mask_motif_overlap --config configs/enhancer_gosai_masked_separator.yaml --ckpt runs/masked_separator/masked_separator_best.ckpt --out_dir runs/masked_separator/analysis/motif_overlap --num_samples 1000 --split val