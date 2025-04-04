#!/usr/bin/env bash
cd /wanghaixin/OmniMol

# 你可以把 REMARK 部分也抽象成可配置变量，如果不同任务只有某一部分需要变动，可以放在循环里更改
# 后面仅加入-combination
BASE_REMARK="temp"
MODEL_BASE_PATH="/wanghaixin/OmniMol/_checkpoints/lora/llama-temp/checkpoint-30"

# ------------------------
# 可自定义的固定参数
# ------------------------
TYPE="partial"
PROMPT="llama3"
BACKBONE="/wanghaixin/OmniMol/checkpoints/Llama-3.2-1B-Instruct"
GRAPH_TOWER="moleculestm"
GRAPH_PATH="/wanghaixin/OmniMol/checkpoints/moleculestm.pth"
BATCH_SIZE=1
DTYPE="bfloat16"
DEVICE="cuda"
MAX_NEW_TOKENS=512
NUM_BEAMS=1
TOP_P=1.0
TEMPERATURE=0.2
REPETITION_PENALTY=1.0
ADD_SELFIES=True
IS_TRAINING=False

# ------------------------
# 需要遍历的 TASK 列表
# ------------------------
# TASK_LIST=(
#      "forward"
#      "reagent"
#      "retrosynthesis"
#      "homolumo"
#      "molcap"
#      "solvent"
#      "catalyst"
#      "yield_BH"
#      "yield_SM"
#     "dqa"
#     "scf"
#     "logp"
#     "weight"
#     "tpsa"
#     "complexity"
#     "experiment"
# )

TASK_LIST=("iupac")



# ------------------------
# 遍历执行
# ------------------------
for TASK in "${TASK_LIST[@]}"; do
    # 根据不同的 TASK 指定 data_path。也可根据需要自行设置更多逻辑。
    case "$TASK" in
        "forward")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/forward_reaction_prediction.json"
            ;;
        "reagent")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/reagent_prediction.json"
            ;;
        "retrosynthesis")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/retrosynthesis.json"
            ;;
        "homolumo")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/property_prediction.json"
            ;;
        "molcap")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/molcap_test.json"
            ;;
        "solvent")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/solvent_pred.json"
            ;;
        "catalyst")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/catalyst_pred.json"
            ;;
        "yield_BH")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/yields_regression_BH.json"
            ;;
        "yield_SM")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/yields_regression_SM.json"
            ;;
        "dqa")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "scf")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "logp")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "weight")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "tpsa")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "complexity")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "iupac")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/selfies2iupac.json"
            ;;
        "experiment")
            DATA_PATH="/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/exp_procedure_pred.json"
            ;;
        *)
            echo "Warning: Unknown TASK: $TASK"
            continue
            ;;
    esac

    # 根据任务生成带有任务名称的 REMARK
    REMARK="${BASE_REMARK}-${TASK}"
    if [ "$TASK" == "yields_regression_BH" ]; then
    MODEL_REMARK="${BASE_REMARK}-yields_regression"
    elif [ "$TASK" == "yields_regression_SM" ]; then
    MODEL_REMARK="${BASE_REMARK}-yields_regression"
    else
    MODEL_REMARK=$REMARK
    fi

    # 输出文件路径
    PPP_PATH="/wanghaixin/OmniMol/eval_result/save_all_tasks/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-${REMARK}-answer.json"
    # 模型路径
    # MODEL_PATH="${MODEL_BASE_PATH}/${MODEL_REMARK}"
    MODEL_PATH="${MODEL_BASE_PATH}"
    # 输出性能的路径
    METRIC_PATH="/wanghaixin/OmniMol/eval_result/save_all_tasks_metric/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-${REMARK}-metric.json"


    # 打印提示信息，方便调试
    echo "--------------------------------------"
    echo " Running TASK:         $TASK"
    echo " Data file:            $DATA_PATH"
    echo " Model path:           $MODEL_PATH"
    echo " Metric path           $METRIC_PATH"
    echo " Output file:          $PPP_PATH"
    echo "--------------------------------------"

    # 执行命令
    # export NLTK_DATA=/root/autodl-tmp/nltk_data
    /root/anaconda3/bin/deepspeed eval_engine.py \
        --model_type "$TYPE" \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --metric_path "$METRIC_PATH" \
        --language_backbone "$BACKBONE" \
        --prompt_version "$PROMPT" \
        --graph_tower "$GRAPH_TOWER" \
        --graph_path "$GRAPH_PATH" \
        --num_beams "$NUM_BEAMS" \
        --top_p "$TOP_P" \
        --temperature "$TEMPERATURE" \
        --data_path "$DATA_PATH" \
        --output_path "$PPP_PATH" \
        --batch_size "$BATCH_SIZE" \
        --dtype "$DTYPE" \
        --use_flash_atten True \
        --device "$DEVICE" \
        --add_selfies "$ADD_SELFIES" \
        --is_training "$IS_TRAINING" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --repetition_penalty "$REPETITION_PENALTY"

    echo "Finish TASK: $TASK"
    echo
done

echo "All tasks finished!"
