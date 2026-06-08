#!/bin/bash
#SBATCH --job-name=measure_NABC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=00:29:00
#SBATCH --qos=standby
#SBATCH --array=0-99
#SBATCH --output=NABC/log/measure_NABC_%A_%a.out
#SBATCH --error=NABC/log/measure_NABC_%A_%a.err


# Create the output_log directory if it doesn't exist
mkdir -p NABC/log

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NPE_ABC
cd $SLURM_SUBMIT_DIR

# -------------------------------------------------------
# Array index encodes (x0_ind, seed):
#   SLURM_ARRAY_TASK_ID = x0_ind * 10 + (seed - 1)
#   x0_ind in {0,...,9}, seed in {1,...,10}
# -------------------------------------------------------

X0_IND=$(( SLURM_ARRAY_TASK_ID / 10 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 10 + 1 ))

echo "Job array ID : $SLURM_ARRAY_TASK_ID"
echo "x0_ind       : $X0_IND"
echo "seed         : $SEED"

python utils/get_measure_NABC.py \
    --task        "mog_10_nabc"  \
    --measure     "c2st"        \
    --x0_ind      "$X0_IND"     \
    --seed        "$SEED"       \
    --post_n_samples  10000     \
    --num_training    1500000      \
    --cond_den    "nsf"         \
    --method      "FMPE"