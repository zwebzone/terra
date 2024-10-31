# ERM
CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_ERM --dataset OfficeHome --data_dir dataset --trial_seed 0 --algorithm ERM --checkpoint_freq 100 --lr 1e-5 --weight_decay 1e-6 --resnet_dropout 0.5 --swad False

# MIRO
CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_MIRO --algorithm MIRO --dataset OfficeHome --data_dir dataset --trial_seed 0 --checkpoint_freq 100 --lr 3e-5 --weight_decay 1e-6 --resnet_dropout 0.1 --swad False

# ERM+Terra
CUDA_VISIBLE_DEVICES=0 python train_extra.py OfficeHome_ERM_Terra --dataset OfficeHome --data_dir dataset --trial_seed 0 --algorithm ERM --checkpoint_freq 100 --lr 1e-5 --weight_decay 1e-6 --resnet_dropout 0.5 --swad False --extra_dir generated_datadir --test_envs 0 1 2 3

# SWAD+Terra
CUDA_VISIBLE_DEVICES=0 python train_extra.py OfficeHome_SWAD_Terra --dataset OfficeHome --data_dir dataset --trial_seed 0 --algorithm ERM --checkpoint_freq 100 --lr 5e-5 --weight_decay 0 --resnet_dropout 0 --swad LossValley --extra_dir generated_datadir --test_envs 0 1 2 3

# SAGM+Terra
CUDA_VISIBLE_DEVICES=0 python train_extra.py OfficeHome_SAGM_Terra --dataset OfficeHome --data_dir dataset --trial_seed 0 --algorithm SAGM_DG --checkpoint_freq 100 --alpha 0.0005 --lr 1e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False --extra_dir generated_datadir --test_envs 0 1 2 3



