gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
# get gpu number
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export HF_HOME="/scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/group6/models"
python -m scripts.generate_embedding_webdataset --cfg-path configs/qwen2_vl_embed_ccsbu.yaml