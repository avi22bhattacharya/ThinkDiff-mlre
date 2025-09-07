gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
# get gpu number
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun --nproc-per-node $gpu_num --rdzv_endpoint=localhost:10000 -m train --cfg-path train_configs/qwen2_vl_vllm_mi_embed_decoder/qwen2_vl_vllm_mi_embed_decoder_ccsub_1.yaml
