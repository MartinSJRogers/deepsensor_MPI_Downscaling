#!/bin/bash 
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --job-name=ds_test_01
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=23:30:30
#SBATCH --mem=250G 
#SBATCH --cpus-per-task=16 
#SBATCH --qos=orchid



source ~/.bashrc

conda activate /home/users/marrog/miniforge-pypy3/envs/ds42_v2_ipy

cd /home/users/marrog/palaeoclim/mrogers

export LD_LIBRARY_PATH=/home/users/marrog/miniforge-pypy3/envs/ds42_v2_ipy/lib:$LD_LIBRARY_PATH

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
which nvidia-smi && nvidia-smi || echo "nvidia-smi not found or GPU not available"
python -c "import torch; print('Torch sees CUDA:', torch.cuda.is_available())"

python HCLIM_perfect_downscale_config.py
