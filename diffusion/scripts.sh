export MODEL_NAME="/data/models/stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR=""
export VALID_PROMPT="A [CLASS]"
export VAE_NAME="/data/models/madebyollin/sdxl-vae-fp16-fix"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_Terra_DG.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --pretrained_vae_model_name_or_path=$VAE_NAME \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir=$OUTPUT_DIR \
 --instance_prompt="${PROMPT}" \
 --rank=32 \
 --resolution=1024 \
 --train_batch_size=8 \
 --learning_rate=5e-5 \
 --report_to="wandb" \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=5000 \
 --validation_prompt="${VALID_PROMPT}" \
 --validation_epochs=1 \
 --seed="0" \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --theta_weight=0.0 \
 --randomimage=0.0 \
 --lr_theta=100



export MODEL_NAME="/data/models/stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR=""
export VALID_PROMPT="A [CLASS]"
export VAE_NAME="/data/models/madebyollin/sdxl-vae-fp16-fix"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_Terra_UDA.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --pretrained_vae_model_name_or_path=$VAE_NAME \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir=$OUTPUT_DIR \
 --instance_prompt="${PROMPT}" \
 --rank=32 \
 --resolution=1024 \
 --train_batch_size=8 \
 --learning_rate=5e-5 \
 --report_to="wandb" \
 --lr_scheduler="constant" \
 --lr_warmup_steps=0 \
 --max_train_steps=5000 \
 --validation_prompt="${VALID_PROMPT}" \
 --validation_epochs=1 \
 --seed="0" \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --theta_weight=0.0 \
 --randomimage=0.0 \
 --lr_theta=100