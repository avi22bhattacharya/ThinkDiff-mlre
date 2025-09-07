gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
# get gpu number
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
cfg=$2

export MASTER_ADDR=localhost
export MASTER_PORT=10000

python -m scripts.test.test_blip_vision_t5_decoder_flux --cfg-path $cfg

# bash runs/test_thinkdiff_clip_two_images.sh 0 configs/test_thinkdiff_clip_two_images.yaml