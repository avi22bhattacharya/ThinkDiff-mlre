gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
# get gpu number
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun --nproc-per-node $gpu_num --rdzv_endpoint=localhost:9998 -m scripts.test.test_mllama_t5_decoder_flux --cfg-path configs/test_thinkdiff_lvlm_ccsbu_image_text.yaml
