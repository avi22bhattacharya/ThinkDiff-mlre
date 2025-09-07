gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
# get gpu number
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python -m scripts.generate_embedding_webdataset --cfg-path configs/qwen2_vl_generate_ccsub.yaml