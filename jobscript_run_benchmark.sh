#BSUB -J run_benchmark
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o run_benchmark.%J.out
#BSUB -e run_benchmark.%J.err
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


# === Run baselines ===
# python -m scripts.run_drakes_baseline --config configs/drakes_baseline.yaml --out_dir ./runs/drakes_baseline
python -m scripts.run_dna_diffusion_baseline --config configs/dna_diffusion_baseline.yaml --out_dir ./runs/dna_diffusion_baseline
# python -m scripts.run_svdd_baseline --config configs/svdd_baseline.yaml --out_dir ./runs/svdd_baseline
# python -m scripts.run_ctrl_dna_baseline --config configs/ctrl_dna_baseline.yaml --out_dir ./runs/ctrl_dna_baseline
# python -m scripts.run_guided_sampling --config configs/guided_sampling.yaml --out_dir ./runs/guided

# === Evaluate baselines ===
ORACLE_CKPT=/nfs/team361/dj16/projects/sequence_generation_baselines/DRAKES_data/data_and_model/mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt
TRAIN=/lustre/scratch126/cellgen/lotfollahi/dj16/data/sequence_data/enhancer_gosai_DRAKES/processed_data/gosai_all.csv

for csv in runs/drakes_baseline/drakes_sequences.csv \
           runs/svdd_baseline/svdd_sequences.csv \
           runs/dna_diffusion_baseline/dna_diffusion_sequences.csv \
           runs/guided/bounded/guided_sequences.csv; do
    out=$(dirname $csv)/eval
    python -m scripts.evaluate_sequences \
      --generated $csv \
      --train_data $TRAIN \
      --scorer_ckpt runs/ensemble/ensemble_best.ckpt \
      --oracle_ckpt $ORACLE_CKPT \
      --oracle_target_idx 0 \
      --jaspar \
      --out_dir $out
done