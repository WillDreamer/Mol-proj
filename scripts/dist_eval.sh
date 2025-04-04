TYPE=lora
PROMPT=llama
MODEL_PATH=/wanghaixin/OmniMol/_checkpoints/moe/conflict-llava-moleculestm-llama-naive_linear-deepseek-moe-3expert-second-half/checkpoint-46280
BACKBONE=/wanghaixin/OmniMol/checkpoints/Llama-3.2-1B-Instruct
REMARK=deepseek-moe-3expert-second-half-46280

# TASK=fwd_pred
# DATA=/wanghaixin/OmniMol/Molecule-oriented_Instructions/forward_reaction_prediction.json
# TASK=reag_pred
# DATA=/wanghaixin/OmniMol/Molecule-oriented_Instructions/reagent_prediction.json
# TASK=retrosyn
# DATA=/wanghaixin/OmniMol/Molecule-oriented_Instructions/retrosynthesis.json
# TASK=prop_pred
# DATA=/wanghaixin/OmniMol/Molecule-oriented_Instructions/property_prediction.json
TASK=molcap
DATA=/wanghaixin/OmniMol/Molecule-oriented_Instructions/molcap_test.txt





ANSWER_PATH=eval_result/$TASK-$TYPE-$PROMPT-$REMARK-answer.json
/root/anaconda3/bin/accelerate launch --main_process_port 29505 eval_engine.py \
    --model_type $TYPE \
    --task $TASK \
    --model_path $MODEL_PATH \
    --language_backbone $BACKBONE \
    --prompt_version $PROMPT \
    --graph_tower moleculestm \
    --graph_path /wanghaixin/OmniMol/checkpoints/moleculestm.pth \
    --num_beams 1 \
    --top_p 1.0 \
    --temperature 0.2 \
    --data_path $DATA \
    --output_path $ANSWER_PATH \
    --batch_size 1 \
    --dtype bfloat16 \
    --use_flash_atten True \
    --device cuda \
    --add_selfies True \
    --is_training False \
    --max_new_tokens 512 \
    --repetition_penalty 1.0