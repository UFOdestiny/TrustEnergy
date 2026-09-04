#!/usr/bin/env bash
#SBATCH --job-name=trustenergy
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Usage:  sbatch jobs/train.sh
#         DATASETS="chicago_15min nyc_manhattan_15min" YEARS=2018 sbatch jobs/train.sh
set -o pipefail
date

module load cuda conda
conda activate st

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
YEARS=${YEARS:-2018}
BS=${BS:-64}
PROJ=${PROJ:-TrustEnergy}
EXTRA=${EXTRA:-}
DATASETS=${DATASETS:-"chicago_15min nyc_manhattan_15min"}

for dataset in $DATASETS; do
    echo "=== TrustEnergy on $dataset ==="
    "$PYTHON" "$REPO/src/flow/trustenergy/main.py" \
        --dataset "$dataset" --years "$YEARS" --bs "$BS" --proj "$PROJ" $EXTRA
    echo "exit code $?"
done
date
