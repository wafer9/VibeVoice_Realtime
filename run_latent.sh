#!/bin/bash

. ./path.sh || exit 1;

export OMP_NUM_THREADS=1

# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # bond0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0  # 1 金山机器
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=2
export NCCL_PXN_DISABLE=0

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
echo "Parsed device_ids: ${device_ids[@]}"

stage=2
stop_stage=2

num_nodes=$1
rank=$2

echo ${num_nodes} ${rank}

dir=exp/vibe_base_cfg_conv_12
tensorboard_dir=${dir}/tensorboard
num_workers=8
prefetch=10

train_engine=torch_ddp # torch_ddp deepspeed
train_config=conf/run_stage1_d2v_12.yaml


. tools/parse_options.sh || exit 1;

set -u
set -o pipefail

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Start pre-training"
fi

# train_set="zh_other_part3"
# shards_dir=/nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg/data/emilia_latent/shard_en
# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#   echo "Making shards, please wait..."
#   for x in ${train_set}; do
#     mkdir -p ${shards_dir}
#     python tools/make_shard_stage_latent.py \
#       --gpus 0,1,2,3,4,5,6,7 \
#       --prefix ${x}\
#       --vibevoice_path /nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B \
#       data/emilia_latent/${x}.list \
#       ${shards_dir} \
#       data/emilia_latent/${x}_shard.list
#   done
# fi


# en：4 台机器分别 node_rank=0/1/2/3，每台 8 卡
# python tools/extract_semantic_latent.py \
#   --gpus 0,1,2,3,4,5,6,7 --num_nodes ${num_nodes} --node_rank ${rank} \
#   --audio_root /nfs-speech-cfs/ASR/opensource/Emilia/ZH/ \
#   --vibevoice_path /nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B \
#   /nfs-speech-cfs/wangzhou/s2s/TwinLakes_vibevoice/data/emilia/zh.jsonl \
#   /nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg/data/emilia_latent/zh_semantic_latent.jsonl

# python tools/extract_semantic_latent.py --merge --world_size 32 \
#   /nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg/data/emilia_latent/en_semantic_latent.jsonl


python tools/make_shard_latent_sim.py  --with_spk_emb