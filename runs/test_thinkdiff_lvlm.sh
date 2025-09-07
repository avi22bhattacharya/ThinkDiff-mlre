gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
# get gpu number
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun --nproc-per-node $gpu_num --rdzv_endpoint=localhost:9999 -m scripts.test.test_mllama_t5_decoder_flux_multi_image --cfg-path configs/qwen2_vl_vllm_mi_embed_decoder_ccsub_1_generate.yaml