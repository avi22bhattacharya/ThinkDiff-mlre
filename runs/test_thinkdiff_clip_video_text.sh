gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id
# get gpu number
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
cfg=$2

python -m scripts.test.test_blip_vision_t5_decoder_cogvideo --cfg-path $cfg

# bash runs/test_thinkdiff_clip_video_text.sh 0 configs/test_thinkdiff_clip_video_text.yaml