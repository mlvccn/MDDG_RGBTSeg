#!/bin/bash -l

#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --mail-user=emailaddr@domain.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -p batch
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged
#SBATCH --gres=gpu:2



# TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node 4 --master_port 25641  tools/train_ddp.py --cfg nyu_rgbd.yaml 
# torchrun --nproc_per_node 4  tools/train_ddp.py --cfg configs/pst_rgbt.yaml
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node 4  tools/train_ddp.py --cfg configs/mf_rgbt.yaml  