#!/bin/bash
#SBATCH --open-mode=truncate
#SBATCH --parsable
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.7.4
source /home/al311/survivenv/bin/activate

python -u ../scripts/PWA_scalars_age_and_survival.py $1 $2 $3 && echo "PYTHON SCRIPT COMPLETED"

