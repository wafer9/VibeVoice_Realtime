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

stage=3
stop_stage=3

num_nodes=$1
rank=$2
train_set="zh"
echo ${num_nodes} ${rank}

dir=exp/s0
tensorboard_dir=${dir}/tensorboard
num_workers=8
prefetch=10

train_engine=torch_ddp # torch_ddp deepspeed
train_config=conf/run_stage1_latent.yaml
shard_dir=/data/joe/data/Emilia_latent_shards/train
shard_list=/data/joe/data/Emilia_latent_shards/train.list


# . tools/parse_options.sh || exit 1;

set -u
set -o pipefail

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Start pre-training"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Making shards, please wait..."
  python tools/make_shard.py \
    --latent-root /data/joe/data/latent \
    --executor process \
    --shuffle --seed 20261028 \
    --shuffle-temp-dir /data/joe/data/Emilia_latent_shards \
    --shuffle-memory 1G \
    --num_utts_per_shard 10000  \
    --num_threads 64 \
    data/Emilia/train.jsonl \
    "${shard_dir}" \
    "${shard_list}" || exit 1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start finetune"
  mkdir -p ${dir}/log
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  dist_backend="nccl"
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes \
           --nproc_per_node=$num_gpus \
           --node_rank=$rank \
           --master_addr="10.126.203.172" \
           --master_port=54322 \
    twinlakes/bin/train.py \
      --config $train_config \
      --data_type "shard" \
      --train_data "${shard_list}" \
      --model_dir $dir \
      --num_workers ${num_workers} \
      --tensorboard_dir $dir/tensorboard \
      --pin_memory > ${dir}/log/${rank}.log 2>&1 
fi
