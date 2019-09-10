#!/bin/bash
#SBATCH -J shaolun-huang-experiment-by-feng
#SBATCH --ntasks=6
python optimization.py --collect --sample_times=800 --train_times=500 --n=6 --k=2

