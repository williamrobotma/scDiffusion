curl https://zenodo.org/records/8286452/files/annotation_model_v1.tar.gz -o output/pretrained/annotation_model_v1.tar.gz --create-dirs
tar -C output/pretrained/ -xvf output/pretrained/annotation_model_v1.tar.gz
rm output/pretrained/annotation_model_v1.tar.gz

cd VAE
python VAE_train.py --data_dir "../../diffusion-scratch/data/processed/zeng23/zeng23_hvg.h5ad" --train_split_only --save_dir '../output/checkpoint/AE/zeng23_hvg' --max_steps 200000 --state_dict ../output/pretrained/annotation_model_v1 |& tee -a train.log
# python VAE_train.py --data_dir "../../diffusion-scratch/data/processed/zeng23/zeng23_full.h5ad" --train_split_only --save_dir '../output/checkpoint/AE/zeng23_full' --max_steps 200000 --state_dict ../output/pretrained/annotation_model_v1 |& tee -a train.log

cd ..
mkdir output/checkpoint/backbone
python cell_train.py --data_dir '../diffusion-scratch/data/processed/zeng23/zeng23_hvg.h5ad' --train_split_only --vae_path 'output/checkpoint/AE/zeng23_hvg/model_seed=0_step=150000.pt' --model_name 'zeng23_hvg' --save_dir 'output/checkpoint/backbone' --lr_anneal_steps 800000 --device_ids cuda:0 |& tee -a train_backbone.log
# python cell_train.py --data_dir '../diffusion-scratch/data/processed/zeng23/zeng23_full.h5ad' --train_split_only --vae_path 'output/checkpoint/AE/zeng23_full/model_seed=0_step=150000.pt' --model_name 'zeng23_full' --save_dir 'output/checkpoint/backbone' --lr_anneal_steps 800000 --device_ids cuda:0 |& tee -a train_backbone.log

mkdir output/checkpoint/classifier
python classifier_train.py --data_dir '../diffusion-scratch/data/processed/zeng23/zeng23_hvg.h5ad' --train_split_only --model_path "output/checkpoint/classifier/zeng23_hvg" --iterations 400000 --vae_path 'output/checkpoint/AE/zeng23_hvg/model_seed=0_step=150000.pt' |& tee -a train_classifier.log
# python classifier_train.py --data_dir '../diffusion-scratch/data/processed/zeng23/zeng23_full.h5ad' --train_split_only --model_path "output/checkpoint/classifier/zeng23_full" --iterations 400000 --vae_path 'output/checkpoint/AE/zeng23_full/model_seed=0_step=150000.pt' |& tee -a train_classifier.log

mkdir output/zeng23_hvg
python classifier_sample.py --model_path 'output/checkpoint/backbone/zeng23_hvg/model800000.pt' --classifier_path 'output/checkpoint/classifier/zeng23_hvg/model200000.pt' --sample_dir 'output/zeng23_hvg/bm' --num_samples 3000 --batch_size 1000 |& tee -a sample.log
# mkdir output/zeng23_full
# python classifier_sample.py --model_path 'output/checkpoint/backbone/zeng23_full/model800000.pt' --classifier_path 'output/checkpoint/classifier/zeng23_full/model200000.pt' --sample_dir 'output/zeng23_full/bm' --num_samples 3000 --batch_size 1000 |& tee -a sample.log