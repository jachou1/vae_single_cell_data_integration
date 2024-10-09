#!/bin/bash

# Define job-specific options
CONTAINER_PATH="/data1/tanseyw/images/pytorch_latest.sif"
FILEPATH_DIR="/data1/tanseyw/projects/jacqueline_sc/vae_amp_xenium/241006_hyperparam_test"

# List of hyperparameters to tune
RECON_WEIGHTS=(0.01 0.1 10)
CONTRASTIVE_WEIGHTS=(0.01 0.1 10)
KL_WEIGHTS=(0.01 0.1 10)
CLASSIFICATION_WEIGHTS=(0.01 0.1 10)

# Pull the singularity container if it's not already present
if [ ! -f "$CONTAINER_PATH" ]; then
  singularity pull "$CONTAINER_PATH" docker://pytorch/pytorch
fi

for recon_param in "${RECON_WEIGHTS[@]}"; do
  for contrastive_param in "${CONTRASTIVE_WEIGHTS[@]}"; do
    for kl_param in "${KL_WEIGHTS[@]}"; do
      for classifier_param in "${CLASSIFICATION_WEIGHTS[@]}"; do
          JOB_NAME="recon${recon_param}_contrastive${contrastive_param}_kl${kl_param}_class${classifier_param}"
          OUTPUT_DIR="${FILEPATH_DIR}/${JOB_NAME}"
          mkdir -p "$OUTPUT_DIR"

          sbatch -p componc_gpu --job-name="$JOB_NAME" \
                 --output="$OUTPUT_DIR/slurm-%j.out" \
                 --error="$OUTPUT_DIR/slurm-%j.err" \
                 --ntasks=1 \
                 --cpus-per-task=4 \
                 --mem=72G \
                 --time=24:00:00 \
                 --gpus=1 \
                 --wrap="singularity run --nv -B /data1/tanseyw/projects/jacqueline_sc/vae_amp_xenium/:/data_folder $CONTAINER_PATH \
                 python /data_folder/multimodal_archetypal_classifier_w_early_stop_v2_running_avg_losses_hyperparameter_sweep.py \
                 concat_sc_xenium_input.npy scrna_labels_subset_xenium_labels_celltype.npy concat_sc_xenium_data_modality.npy 300 \
                 --input-dir /data_folder/240918_experiments \
                 300 0.001 200 1 90 --output_dir $OUTPUT_DIR \
                 --reconstruction_penalty $recon_param --contrastive_penalty $contrastive_param --classification_penalty $classifier_param \
                 --kl_penalty $kl_param"
      done
    done
  done
done
