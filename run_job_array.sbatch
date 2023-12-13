#!/bin/bash
#SBATCH -A hive-jw254
#SBATCH -J array_baryonic
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH -o reports/array_baryonic_%A_%a.out
#SBATCH --mail-user=vhorvath3@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem-per-cpu=64G
#SBATCH --array=0-103

cd /storage/hive/project/phy-wise/vhorvath3/baryonic-pred-from-dm/scripts

source ~/.bashrc

srun python capture_baryonics.py ${SLURM_ARRAY_TASK_ID}
