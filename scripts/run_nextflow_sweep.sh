#!/bin/bash
# Usage: bash scripts/run_nextflow_sweep.sh <input_dir> <output_dir>

# command line args
input_dir=$1
output_dir=$2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "${SCRIPT_DIR}/.."


NXF_SINGULARITY_HOME_MOUNT=true nextflow run main.nf \
	--input_dir ${input_dir} \
	--device cuda \
	--outdir ${output_dir} \
	-w ${output_dir}/nf \
	--max_memory 64GB \
	-profile iris \
	--wandb_api_key "${WANDB_API_KEY}" \
	--max_time 96h \
	--device cuda