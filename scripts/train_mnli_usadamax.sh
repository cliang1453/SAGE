#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train.sh <gpu>"
  exit 1
fi
gpu=$1
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}

modeltype="bert"
modelname="base"
toktype="uncased"
train_datasets="mnli"
test_datasets="mnli_matched,mnli_mismatched"
model_root="/root/data/mtdnn_ckpt"
bert_path="${model_root}/${modeltype}_model_${modelname}_${toktype}.pt"
data_dir="/root/data/mtdnn/canonical_data/${modeltype}_${modelname}_${toktype}_lower"
seed=2018

batch_size=32
batch_size_eval=32
grad_acc_steps=1
answer_opt=1
optim="usadamax"
grad_clipping=0
global_grad_clipping=1
lr="2e-4"
encoder_type=1
epochs=3
beta3=0.85

prefix="${train_datasets}_${optim}_${modeltype}_${lr}_${beta3}"
model_dir="${model_root}/adplr/${seed}/${modelname}/${prefix}"
log_file="${model_dir}/log.log"
python train.py \
--data_dir ${data_dir} \
--batch_size ${batch_size} \
--batch_size_eval ${batch_size_eval} \
--output_dir ${model_dir} \
--log_file ${log_file} \
--answer_opt ${answer_opt} \
--optimizer ${optim} --beta3 ${beta3} \
--train_datasets ${train_datasets} \
--test_datasets ${test_datasets} \
--grad_clipping ${grad_clipping} \
--global_grad_clipping ${global_grad_clipping} \
--learning_rate ${lr} \
--init_checkpoint ${bert_path} \
--epochs ${epochs} \
--encoder_type ${encoder_type} \
--seed ${seed}
