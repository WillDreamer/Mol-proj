#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

# PROMPT_VERSION="phi"
# MODEL_VERSION="phi3"

PROMPT_VERSION="tinyllama"
MODEL_VERSION="tinyllama"

# PROMPT_VERSION="llama3"
# MODEL_VERSION="llama3-1b"

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/graphMVP.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="downloads/moleculestm.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="/root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/himol_encoder.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="_checkpoints/basline"
TASK="molcap"
PROJECTOR="naive_linear"
REMARK="exploration"

deepspeed main.py \
     --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe stage2 \
    --model_name_or_path /root/autodl-tmp/MolMoE/downloads/TinyLlama-1.1B-Chat-v1.0 \
    --base_model /root/autodl-tmp/MolMoE/downloads/TinyLlama-1.1B-Chat-v1.0 \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --exploration True \
    --data_path /root/autodl-tmp/MolMoE/data_files/molcap_train.json \
    --mixed_eval_set /root/autodl-tmp/MolMoE/data_files/eval_splitted \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter /root/autodl-tmp/MolMoE/_checkpoints/pretrain/llava-pub_chem-moleculestm-tinyllama-naive_linear-baseline-linear-projector/checkpoint-30695/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/lora/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --stop_epoch 5 \
    --eval_strategy "epoch" \
    --split_eval False \
    --eval_on_start True \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0075 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --moe_enable False \
    --logging_dir /root/tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --is_training True \