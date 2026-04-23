#BSUB -J run_sequence_generation
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o run_sequence_generation.%J.out
#BSUB -e run_sequence_generation.%J.err
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

# === Generate sequences with guided sampling ===
python -m scripts.guided_sampler --config configs/enhancer_gosai_guided.yaml --out_dir runs/guided --num_batches 40
