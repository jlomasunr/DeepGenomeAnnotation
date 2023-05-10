#!/bin/bash
#SBATCH --job-name=train_bilstm_cpu     # create a short name for your job
#SBATCH --nodes=2                       # node count
#SBATCH --ntasks-per-node=15             # total number of tasks per node
#SBATCH --cpus-per-task=16               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=240G                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=7-00:00:00               # total run time limit (D-HH:MM:SS)
#SBATCH --mail-type=all                 # email notifications
#SBATCH -o train_bilstm_gpu.out
#SBATCH --mail-user=jlomas@nevada.unr.edu

export WORKERS = 8  # Set number of data loading workers

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE -$WORKERS))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

conda activate bilstm

srun python DGA.py train \
    --window 20 \
    --hidden 20 \
    --layers 4 \
    --batchsize 320 \
    --workers $WORKERS \
    --lr 0.1 \
    --outfile model_state.pt \
    --epochs 100