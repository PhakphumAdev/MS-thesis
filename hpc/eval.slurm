#!/bin/bash
#SBATCH --account=pr_95_tandon_priority
#SBATCH --job-name=eval_csr
#SBATCH --output=eval_csr.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=pa2497@nyu.edu
module purge

OVERLAY_FILE=/scratch/pa2497/overlay-25GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/pa2497/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

#make sure to have hf login in your environment
singularity exec --nv --bind $SCRATCH \
	    --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
	    /bin/bash -c "source /ext3/env.sh; bash /scratch/pa2497/MS-thesis/hpc/run_eval.sh"

