process TRAIN {
    tag "$meta.id"
    label "${ params.device == 'cuda' ? 'gpu' : 'cpu' }"
    label 'process_high'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/vae_single_cell_integration:' + params.bayestme_version) :
        ('docker.io/jeffquinnmsk/vae_single_cell_integration:' + params.bayestme_version) }"

    input:
    tuple val(meta), path(input_dir), val(recon_param), val(contrastive_param), val(kl_param), val(classifier_param)

    output:
    tuple val(meta), path("${prefix}/recon_param_${recon_param}_contrastive_param_${contrastive_param}_kl_param_${kl_param}_classifier_param_${classifier_param}"), emit: result
    path  "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def wandb_key = params.wandb_api_key == null ? "" : "${params.wandb_api_key}"
    def wandb_mode = params.wandb_api_key == null ? "disabled" : "online"
    def runid = workflow.runName
    """
    export WANDB_DIR="\$(realpath ${prefix}/wandb)"
    export WANDB_DISABLE_GIT=1
    export WANDB_DISABLE_CODE=1
    export WANDB_API_KEY="${wandb_key}"
    export WANDB_MODE="${wandb_mode}"
    export WANDB_PROJECT="vae_single_cell_data_integration"
    export WANDB_RUN_ID="${runid}"
    export WANDB_CACHE_DIR="\$(realpath ${prefix}/wandb/.cache)"
    export WANDB_DATA_DIR="\$(realpath ${prefix}/wandb/.data)"

    mkdir "${prefix}/recon_param_${recon_param}_contrastive_param_${contrastive_param}_kl_param_${kl_param}_classifier_param_${classifier_param}"
    train \
        --cells-file smaller_fidelitous_input.npy \
        --labels-file smaller_fidelitous_labels.npy \
        --modality-file smaller_fidelitous_modalities.npy \
        --n-epochs 300 \
        --input-dir "$input_dir" \
        --learning-rate 0.001 \
        --batch-size 200 \
        --label-pct 0.1 \
        --output_dir "${prefix}/recon_param_${recon_param}_contrastive_param_${contrastive_param}_kl_param_${kl_param}_classifier_param_${classifier_param}" \
        --reconstruction_penalty ${recon_param} \
        --contrastive_penalty ${contrastive_param} \
        --classification_penalty ${classifier_param} \
        --kl_penalty ${kl_param} \
        --device ${params.device} \
    """
}
