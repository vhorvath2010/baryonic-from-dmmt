#!/bin/bash
#SBATCH -A hive-jw254
#SBATCH -J clean_subhalos
#SBATCH -N1
#SBATCH -phive
#SBATCH -t 2-00:00:00
#SBATCH -o report-%j.out
#SBATCH --mail-user=vhorvath3@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem-per-cpu=64G

cd /storage/hive/project/phy-wise/vhorvath3/baryonic-pred-from-dm/scripts

source ~/.bashrc

srun python clean_non_SM_halos.py
