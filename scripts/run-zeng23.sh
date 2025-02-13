#!/usr/bin/env bash

# set -u
set -x

NAME="zeng23_full"
# NAME="zeng23_hvg"
CUDA_INDEX=0
MODEL_NAME="zeng23_full"
NO_LSN=false
NO_LSN_FLAG=""

while getopts "n:uc:" flag; do
 case $flag in
   n) # Handle the -h flag
   NAME=$OPTARG
   ;;
   u)
   NO_LSN=true
   ;;
   c) # Handle the -v flag
   CUDA_INDEX=$OPTARG
   ;;
   \?)
   # Handle invalid options
   ;;
 esac
done

if [ "$NO_LSN" = true ]; then
    MODEL_NAME="${NAME}_nolsn"
    NO_LSN_FLAG="--nolsn"
fi

if [ ! -d "output/pretrained/annotation_model_v1" ]; then
    curl https://zenodo.org/records/8286452/files/annotation_model_v1.tar.gz -o output/pretrained/annotation_model_v1.tar.gz --create-dirs
    tar -C output/pretrained/ -xvf output/pretrained/annotation_model_v1.tar.gz
    rm output/pretrained/annotation_model_v1.tar.gz
fi

mkdir -p "logs/${MODEL_NAME}"

cd VAE
python VAE_train.py --data_dir "../../diffusion-scratch/data/processed/zeng23/${NAME}.h5ad" --train_split_only $NO_LSN_FLAG --save_dir "../output/checkpoint/AE/${MODEL_NAME}" --max_steps 200000 --state_dict ../output/pretrained/annotation_model_v1 --device "cuda:${CUDA_INDEX}" |& tee -a "../logs/${MODEL_NAME}/train_vae.log"

cd ..
mkdir -p output/checkpoint/backbone
python cell_train.py --data_dir "../diffusion-scratch/data/processed/zeng23/${NAME}.h5ad" --train_split_only $NO_LSN_FLAG --vae_path "output/checkpoint/AE/${NAME}/model_seed=0_step=150000.pt" --model_name "${MODEL_NAME}" --save_dir 'output/checkpoint/backbone' --lr_anneal_steps 800000 --device_ids "cuda:${CUDA_INDEX}" |& tee -a "logs/${MODEL_NAME}/train_backbone.log"

mkdir -p output/checkpoint/classifier
python classifier_train.py --data_dir "../diffusion-scratch/data/processed/zeng23/${NAME}.h5ad" --train_split_only $NO_LSN_FLAG --model_path "output/checkpoint/classifier/${MODEL_NAME}" --iterations 400000 --vae_path "output/checkpoint/AE/${MODEL_NAME}/model_seed=0_step=150000.pt" --device_ids "cuda:${CUDA_INDEX}" |& tee -a "logs/${MODEL_NAME}/train_classifier.log"

mkdir -p "output/${MODEL_NAME}"
python generate_classifer_h5ad.py --data_dir "../diffusion-scratch/data/processed/zeng23/${NAME}.h5ad" --vae_path "output/checkpoint/AE/${MODEL_NAME}/model_seed=0_step=150000.pt" --model_path "output/checkpoint/backbone/${MODEL_NAME}/model800000.pt" --classifier_path "output/checkpoint/classifier/${MODEL_NAME}/model200000.pt" --sample_dir "output/${MODEL_NAME}" --batch_size 1024 --device "cuda:${CUDA_INDEX}" |& tee -a "logs/${MODEL_NAME}/generate_h5ad.log"
# python -p classifier_sample.py --model_path 'output/checkpoint/backbone/${NAME}/model800000.pt' --classifier_path 'output/checkpoint/classifier/${NAME}/model200000.pt' --sample_dir 'output/${NAME}/bm' --num_samples 3000 --batch_size 1000 |& tee -a "logs/${NAME}/sample.log"
