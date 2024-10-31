
# ERM+Terra
CUDA_VISIBLE_DEVICES=0 python train_extra.py VLCS_ERM_Terra --dataset VLCS --data_dir dataset --trial_seed 0 --algorithm ERM --checkpoint_freq 100 --lr 1e-5 --weight_decay 1e-4 --resnet_dropout 0.1 --swad False --extra_dir generated_datadir --test_envs 0 1 2 3
# SWAD+Terra
CUDA_VISIBLE_DEVICES=0 python train_extra.py VLCS_LoRA_SWAD_Terra --dataset VLCS --data_dir dataset --trial_seed 0 --algorithm ERM --checkpoint_freq 100 --lr 5e-5 --weight_decay 0 --resnet_dropout 0 --swad LossValley --extra_dir generated_datadir --test_envs 0 1 2 3

# SAGM
CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_SAGM --dataset VLCS --data_dir dataset --trial_seed 2 --algorithm SAGM_DG --checkpoint_freq 100 --alpha 0.003 --lr 1e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False

# SAGM+Terra
CUDA_VISIBLE_DEVICES=0 python train_extra.py VLCS_LoRA_SAGM_Terra --dataset VLCS --data_dir dataset --trial_seed 0 --algorithm SAGM_DG --checkpoint_freq 100 --alpha 0.003 --lr 1e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False --extra_dir generated_datadir --test_envs 0 1 2 3
