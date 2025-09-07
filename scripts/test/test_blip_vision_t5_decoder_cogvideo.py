"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import thinkdiff.tasks as tasks
from thinkdiff.common.config import Config
from thinkdiff.common.dist_utils import get_rank, init_distributed_mode
from thinkdiff.common.logger import setup_logger
from thinkdiff.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from thinkdiff.common.registry import registry
from thinkdiff.common.utils import now

# imports modules for registration
from thinkdiff.datasets.builders import *
from thinkdiff.models import *
from thinkdiff.processors import *
from thinkdiff.runners import *
from thinkdiff.tasks import *

from PIL import Image
import requests
from transformers import Blip2Processor, T5EncoderModel, T5TokenizerFast, CLIPVisionModel, AutoProcessor
import torch
from thinkdiff.models.flux_prompt import FluxPipelineRewritePrompt
import os
from torch import nn
import re
from torch.nn import functional as F

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from datetime import datetime
import os

def save_video(tensor, video_path):
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # video_path = f"./cogvideo_output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    model = model.eval().to(cfg.run_cfg.device, torch.bfloat16)

    # diffusion_pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b").to("cuda", torch.bfloat16)
    diffusion_pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b").to("cuda", torch.bfloat16)
    diffusion_pipe.transformer.to(memory_format=torch.channels_last)
    diffusion_pipe.transformer = torch.compile(diffusion_pipe.transformer, mode="max-autotune", fullgraph=True)

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")

    output_dir = cfg.run_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if cfg.run_cfg.get("img_urls") is not None:
        img_urls = cfg.run_cfg["img_urls"]
    else:
        img_urls = []
        img_urls += [
            "assets/dreambench_plus_style_11.jpg",
            ]
    
    if cfg.run_cfg.get("image_names") is not None:
        image_names = cfg.run_cfg["image_names"]
    else:
        image_names = []
        image_names += [
            "dreambench_plus_style_11",
            ]

    print(img_urls)
        
    if cfg.run_cfg.get("questions") is not None:
        questions = cfg.run_cfg["questions"]
    else:
        questions = []
        questions += ["A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight casts a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."]

    if cfg.run_cfg.get("names") is not None:
        names = cfg.run_cfg["names"]
    else:
        names = []
        names += ["panda_guitar_1"]

    for img_i, img_url in enumerate(img_urls):
        for prompt_i, prompt in enumerate(questions):
            raw_image = Image.open(img_url)
            name = names[prompt_i]
            image_name = image_names[img_i]

            inputs = processor(raw_image, prompt, return_tensors="pt").to("cuda", torch.bfloat16)
            with torch.no_grad():

                input_vision_token_num = 65

                vision_outputs = model.forward_encoder(
                    **inputs,
                )

                # del model

                t5_prompt_embeds = vision_outputs[:, :input_vision_token_num, :]

                prompt_embeds, _ = \
                    diffusion_pipe.encode_prompt(
                        prompt=prompt,
                        max_sequence_length=226 - input_vision_token_num
                    )
                
                prompt_embeds = torch.cat([t5_prompt_embeds, prompt_embeds], dim=1)


            video = diffusion_pipe(
                prompt_embeds=prompt_embeds.to(torch.bfloat16), 
                guidance_scale=6, 
                num_inference_steps=50).frames[0]
            
            video_path = os.path.join(output_dir, image_name + "_" + name + ".mp4")
            save_video(video, video_path)
            print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    main()
