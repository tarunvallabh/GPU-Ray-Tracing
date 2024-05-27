#!/bin/bash

#SBATCH --job-name=report
#SBATCH --output=slurm_out/%j-report.out
#SBATCH --error=slurm_out/%j-report.err
#SBATCH --account=mpcs51087
#SBATCH --time=02:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=16
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tvallabh@uchicago.edu
#SBATCH --chdir=/home/tvallabh/project-2-winter-2024-tarunvallabh

./parallel 16 1000000000 1000
./parallel 16 1000000000 100



