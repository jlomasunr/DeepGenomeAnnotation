#!/bin/bash
#SBATCH --job-name=train_bilstm_cpu     # create a short name for your job
#SBATCH --nodes=3                       # node count
#SBATCH --ntasks-per-node=15             # total number of tasks per node
#SBATCH --cpus-per-task=4               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=208G                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=14-00:00:0               # total run time limit
#SBATCH --mail-type=all                 # email notifications
#SBATCH -o train_bilstm_cpu.out
#SBATCH --mail-user=jlomas@nevada.unr.edu
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0

#export WORKERS=6  # Set number of data loading workers

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE )) #- $WORKERS))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python DGA.py train Ath_Test.fa BaseWiseClasses.fa \
    --window 20 \
    --hidden 20 \
    --layers 2 \
    --batchsize 64 \
    --workers 0 \
    --lr 0.1 \
    --outfile model_state.pt \
    --epochs 100 \
    --nocuda
