import os
from PIL import Image
import webdataset as wds
from thinkdiff.datasets.datasets.base_dataset import BaseDataset
import random
from torch.utils.data import DataLoader, DistributedSampler
import wids
import json
import tqdm

llava_brief_instructions = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
    "Generate a prompt that can recreate the image in a 2D diffusion model.",
    "Provide a descriptive prompt to reproduce the given image using a diffusion model.",
    "Create a prompt suitable for a 2D diffusion model to generate the same image.",
    "Summarize the visual details as a prompt for a 2D diffusion model.",
    "Write a clear prompt to guide a 2D diffusion model in recreating the image.",
    # "Give a diffusion model prompt to reproduce the image accurately.",
    # "Formulate a prompt based on the image for use in 2D diffusion generation.",
    # "Provide a clear and specific prompt for a 2D diffusion model to generate the picture.",
    # "Render a prompt that can recreate the photo using a 2D diffusion model.",
    # "Write a succinct but informative prompt for a 2D diffusion model to reproduce the image."
]


class CCSBUMllamaVllmProcessDatasetWids(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        # self.inner_dataset = wids.ShardListDataset(location[0], keep=True, cache_dir=location[1])
        self.inner_dataset = wids.ShardListDataset(location, keep=True, localname=lambda x: x)

    def collater(self, samples):
        images = []
        answers = []
        jsons = []
        filenames = []

        for sample in samples:
            images.append([sample[".jpg"].convert("RGB")])
            prompt = random.choice(llava_brief_instructions)
            answers.append(prompt)
            sample[".json"]["prompt"] = prompt
            jsons.append(sample[".json"])
            filenames.append(sample["__key__"])
        
        inputs = {
            "images": images,
            "answers": answers,
            "jsons": jsons,
            "filenames": filenames
        }
        return inputs

