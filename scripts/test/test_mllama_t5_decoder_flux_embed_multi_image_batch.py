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
import tqdm
import io
import json
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


    diffusion_id = "black-forest-labs/FLUX.1-dev"


    # diffusion_pipe = FluxPipelineRewritePrompt.from_pretrained(
    #     diffusion_id,
    #     # device_map="balanced",
    #     torch_dtype=torch.bfloat16
    # )

    # diffusion_pipe.enable_model_cpu_offload()

    output_dir = cfg.run_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    img_folder = cfg.run_cfg["image_folder"]
    json_urls = [os.path.join(img_folder, img_name) for img_name in os.listdir(img_folder)]
    json_urls = [img_url for img_url in json_urls if os.path.isfile(img_url) and img_url.endswith(".json")]
    urls = json_urls

    # diffusion_pipe.set_progress_bar_config(disable=False)


    batch_size = cfg.run_cfg["batch_size"]
    image_path_prefix = cfg.run_cfg.get("image_path_prefix", None)
    for batch_i in range(0, len(urls), batch_size):
        sample_list = []
        json_dict_list = []
        answer_list = []
        image_name_list = []
        for img_i in range(batch_i, batch_i+batch_size):
            if img_i >= len(urls):
                break
            url = urls[img_i]
            image_name = url.split("/")[-1].split(".")[0]
            embed_output_path = f"{output_dir}/{image_name}.pth"
            if os.path.exists(embed_output_path):
                print(f"Image already exists at {embed_output_path}")
                continue

            # if ":" in url:
            #     image = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize([224, 224], Image.Resampling.BICUBIC)
            # else:
            #     image = Image.open(url).convert("RGB").resize([224, 224], Image.Resampling.BICUBIC)
            with open(url, "r") as f:
                json_dict = json.load(f)

            text_inputs = json_dict["text_inputs"]
            image_paths = json_dict["image_inputs"]

            if image_path_prefix is not None:
                image_paths_new = []
                for image_path in image_paths:
                    index = image_path.find("cobsat/datasets")
                    image_path = os.path.join(image_path_prefix, image_path[index:])
                    image_paths_new.append(image_path)
                image_paths = image_paths_new

            # shot = len(image_inputs)

            texts = []
            for text_i in range(len(text_inputs)):
                if text_i == 0:
                    texts.append(f"Word {text_i+1}: " + text_inputs[text_i][0:-2] + ", ")
                else:
                    texts.append(f"\n\nWord {text_i+1}: " + text_inputs[text_i][0:-2] + ", ")

            
            placeholders = []
            for image_path_i, image_path in enumerate(image_paths):
                placeholders.append({"type": "text", "text": texts[image_path_i]})
                if cfg.run_cfg.get("max_pixels", None) is not None:
                    placeholders.append({"type": "image", "image": image_path, "max_pixels": cfg.run_cfg["max_pixels"]})
                else:
                    placeholders.append({"type": "image", "image": image_path})

            placeholders.append({"type": "text", "text": texts[-1]})

            answer = cfg.run_cfg["prompt"]
            messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": answer
                    },
                    *placeholders,
                ],
            }]


            prompt = model.mllama_processor.apply_chat_template(messages,
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
            
            sample_list.append(sample)
            json_dict_list.append(json_dict)
            answer_list.append(answer)
            image_name_list.append(image_name)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                embedding_type = cfg.model_cfg.get("embedding_type", "output_embed")
                if getattr(model, "module", None) is not None:
                    language_model_inputs, mllama_generated_text = model.module.get_embed(sample_list, embedding_type=embedding_type, max_new_tokens=128, need_process=False)
                else:
                    language_model_inputs, mllama_generated_text = model.get_embed(sample_list, embedding_type=embedding_type, max_new_tokens=128, need_process=False)
                # language_model_inputs, mllama_generated_text = model.module.get_embed(sample, embedding_type=embedding_type, max_new_tokens=64, min_new_tokens=64)
                
                for sample_i in range(len(sample_list)):

                    image_name = image_name_list[sample_i]
                    json_dict = json_dict_list[sample_i]
                    answer = answer_list[sample_i]
                    print(mllama_generated_text[sample_i])
                    print(language_model_inputs[sample_i].shape)

                    embed_output_path = f"{output_dir}/{image_name}.pth"
                    json_output_path = f"{output_dir}/{image_name}.json"

                    embed = language_model_inputs[sample_i]
                    embed = embed.cpu()
                    buffer = io.BytesIO()
                    torch.save(embed, buffer)  # Clone the tensor here
                    with open(embed_output_path, "wb") as f:
                        f.write(buffer.getvalue())
                    
                    json_dict["generated_text"] = mllama_generated_text[sample_i]
                    json_dict["prompt"] = answer
                    with open(json_output_path, "w") as f:
                        json.dump(json_dict, f, indent=4)

                    print(f"Saved embed to {embed_output_path}")
                    print(f"Saved json to {json_output_path}")

if __name__ == "__main__":
    main()
