#!/bin/bash
#SBATCH --job-name=train_bilstm
#SBATCH --cpus-per-task=64
#SBATCH --mem=240g
#SBATCH --nodes=2
#SBATCH --time=7-15:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=jlomas@nevada.unr.edu
#SBATCH -o train_bilstm.out
#SBATCH --account=cpu-s5-jlomas-0
#SBATCH --partition=cpu-core-0

python DGA.py train Ath_Train.fa Ath_Train.gff \
            --window 20 --hidden 10 --layers 1 --batchsize 1 \
            --workers=0
