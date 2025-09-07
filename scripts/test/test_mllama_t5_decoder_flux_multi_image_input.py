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
from torch.utils.data import DataLoader
from PIL import Image
import requests
from thinkdiff.datasets.datasets.cc_sbu_dataset_mllama_vllm_process import llava_brief_instructions
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from thinkdiff.models.flux_prompt import FluxPipelineRewritePrompt
# import gc
from qwen_vl_utils import process_vision_info


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
    seed = cfg.run_cfg.seed + get_rank()

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    model = runner.model
    # dataset = datasets[list(datasets.keys())[0]]["train"]
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     num_workers=1,
    #     pin_memory=True,
    #     collate_fn=dataset.collater
    # )
    model.eval()

    device = model.device
    use_amp = True
    amp_dtype = torch.bfloat16
    ckpt_id = os.path.basename(cfg.model_cfg["ckpt"])


    diffusion_id = "black-forest-labs/FLUX.1-dev"


    diffusion_pipe = FluxPipelineRewritePrompt.from_pretrained(
        diffusion_id,
        # device_map="balanced",
        torch_dtype=torch.bfloat16
    )

    diffusion_pipe.enable_model_cpu_offload()

    output_dir = cfg.run_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # question = "I give you several words and images. First, please analyse what the missing image is. Then give me a detailed diffusion prompt to describe the missing image. Please only provide me the detailed prompt and start the answer with 'Create an image'.\n\n"
    # question = "I give you several words and images. First, please analyse what the next image is. Then give me a detailed diffusion prompt to describe the next image. Please only provide me the detailed prompt and start the answer with 'Create an image'.\n\n"
    # question = "I give you several words and pictures. First, please analyse what the next picture is. Then give me a detailed diffusion prompt to describe the next picture. Please only provide me the detailed prompt and start the answer with 'Create an image'.\n\n"
    # question = "I give you several words and images. Please give me a detailed diffusion prompt to describe the missing image. Please only provide me the detailed prompt and start the answer with 'Create an image'.\n\n"
    # question = "I give you several words and images. Please give me a detailed diffusion prompt to describe the missing image started with 'Create an image'.\n\n"
    # question = "I give you several words and images. Please give me a detailed diffusion prompt to describe the missing image started with 'Create an image'.\n\nHere are the words and images.\n\n"
    # question = ""
    # question = "Please make the dog as a black dog."
    # question = "The dog is a black dog."
    # question = "The dog is a Chihuahua dog."
    # question = "The dog is a Poodle dog."
    # question = "The dog is a red dog."
    question = "Reconstruct the texts in this image."

    # image_paths = []
    # image_paths.append("/root/dataset/minigpt4/cobsat/datasets/texture_box/denim_box.jpg")
    # image_paths.append("/root/dataset/minigpt4/cobsat/datasets/texture_cup/denim_cup.jpg")

    # texts = []
    # texts.append("box")
    # texts.append("cup")
    # texts.append("apple")

    # image_names = []
    # image_names.append("denim_" + "_".join(texts).replace(" ", "_"))


    # image_paths = []
    # image_paths.append("/root/dataset/minigpt4/cobsat/datasets/texture_box/denim_box.jpg")
    # image_paths.append("/root/dataset/minigpt4/cobsat/datasets/texture_cup/denim_cup.jpg")

    # texts = []
    # texts.append("Word 1: box, ")
    # texts.append("\n\nWord 2: cup, ")
    # texts.append("\n\nWord 3: apple, ")

    # image_names = []
    # # image_names.append("denim_" + "_".join(texts).replace(" ", "_"))
    # image_names.append("denim_box_cup_apple")

    # image_paths = []
    # image_paths.append("/root/dataset/minigpt4/cobsat/datasets/color_car/white_car.jpg")
    # image_paths.append("/root/dataset/minigpt4/cobsat/datasets/color_car/blue_car.jpg")
    # # image_paths.append("/root/dataset/minigpt4/cobsat/datasets/color_car/red_car.jpg")

    # texts = []
    # # texts.append("white: ")
    # # texts.append("blue: ")
    # # texts.append("red: ")
    # texts.append("Word 1: white, ")
    # texts.append("\n\nWord 2: blue, ")
    # texts.append("\n\nWord 3: red, ")

    # image_names = []
    # image_names.append("car_white_blue_red")
    # # image_names.append("car_white_red_blue")



    # image_paths = []
    # image_paths.append("/root/dataset/minigpt4/dreambooth/dataset/dog/00.jpg")
    # image_paths.append("/root/dataset/minigpt4/dreambooth/dataset/pink_sunglasses/01.jpg")


    # texts = []
    # # texts.append("The dog is a Poodle dog.")
    # texts.append("")
    # texts.append("")
    # texts.append("")


    # image_names = []
    # # image_names.append("dog_glass")
    # image_names.append("black_dog")
    # # image_names.append("Chihuahua_dog")
    # # image_names.append("Poodle_dog")
    # # image_names.append("red_dog")




    # image_paths = []
    # image_paths.append("imgs/gordon.jpg")

    # texts = []
    # texts.append("")


    # image_names = []
    # image_names.append("gordon")




    image_paths = []
    image_paths.append("/root/dataset/minigpt4/MARIOEval/MARIOEval/LAIONEval4000/images/0.jpg")

    texts = []
    texts.append("")


    image_names = []
    image_names.append("LAIONEval4000_0")


    


    placeholders = []
    for image_path_i, image_path in enumerate(image_paths):
        placeholders.append({"type": "text", "text": texts[image_path_i]})
        # placeholders.append({"type": "image", "image": image_path})
        placeholders.append({"type": "image", "image": image_path, "max_pixels": 65536})
    
    placeholders.append({"type": "text", "text": texts[-1]})

    # placeholders = [{"type": "image", "image": url} for url in image_urls]


    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            # {
            #     "type": "text",
            #     "text": question
            # },
        ],
    }]

    prompt = model.module.mllama_processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True,
                                           add_vision_id=True)
    print(prompt)

    image_data, _ = process_vision_info(messages)

    sample =  {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        }

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            embedding_type = cfg.model_cfg.get("embedding_type", "output_embed")
            language_model_inputs, mllama_generated_text = model.module.get_embed(sample, embedding_type=embedding_type, max_new_tokens=128, need_process=False)
            # language_model_inputs, mllama_generated_text = model.module.get_embed(sample, embedding_type=embedding_type, max_new_tokens=64, min_new_tokens=64)
            for i in range(len(mllama_generated_text)):
                print(language_model_inputs[i].shape)
                # print(sample["answers"][i])
                print(mllama_generated_text[i])
            pass

    guidance_scale = 3.5
    for img_i in range(1):

        image_name = image_names[img_i]
        with torch.no_grad():

            language_model_inputs_i = language_model_inputs[img_i]
            language_model_inputs_i = language_model_inputs_i.unsqueeze(0)
            prompt_embeds, pooled_prompt_embeds, _ = \
                diffusion_pipe.encode_prompt(
                    prompt="",
                    prompt_2=None,
                    prompt_embeds=language_model_inputs_i
                )
            
            txt_prompt_embeds, pooled_prompt_embeds, _ = \
                diffusion_pipe.encode_prompt(
                    prompt=question,
                    prompt_2=None,
                    max_sequence_length=128
                )
            
            prompt_embeds = torch.cat([prompt_embeds, txt_prompt_embeds], dim=1)

        diffusion_pipe.set_progress_bar_config(disable=False)
        setup_seeds(cfg)

        images = diffusion_pipe(
            prompt_embeds=prompt_embeds.to(torch.bfloat16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
            num_images_per_prompt=1,
            height=512,
            width=512,
            # height=1024,
            # width=1024,
            num_inference_steps=28,
            guidance_scale=guidance_scale,
        ).images

        # output_dir = f"logs/original_blip2_flux_cfg_0"
        for image_i, image in enumerate(images):
            image.save(f"{output_dir}/{image_name}_output_embed_edit_4_flux_{image_i}_{ckpt_id}_seed_{seed}.png", format="PNG", compress_level=1)
            print(f"Saved image to {output_dir}/{image_name}_output_embed_edit_4_flux_{image_i}_{ckpt_id}_seed_{seed}.png")

if __name__ == "__main__":
    main()
