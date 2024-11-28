#!/bin/bash
#SBATCH -A eecs   # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p gpu                            # name of partition or queue
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH -c 1
#SBATCH -t 7-00:00:00
# load any software environment module required for app (e.g. matlab, gcc, cuda)
# run my job (e.g. matlab, python)