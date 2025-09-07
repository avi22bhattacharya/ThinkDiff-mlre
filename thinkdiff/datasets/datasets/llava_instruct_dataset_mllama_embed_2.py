import os
from PIL import Image
import webdataset as wds
from thinkdiff.datasets.datasets.base_dataset import BaseDataset
import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import functional as F

class LlavaInstructMllamaEmbedDataset_2(BaseDataset):
    def __init__(self, vis_processor, text_processor, location, build_info):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.build_info = build_info

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            # wds.to_tuple("jpg", "json", "pth", handler=wds.warn_and_continue),
            # wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    # def getfname(self, sample):
    #     sample["filename"] = sample["__key__"]

    # def to_dict(self, sample):
    #     return {
    #         "image": sample[0],
    #         "json": sample[1],
    #         "filename": sample[2]
    #     }

    def collater(self, samples):
        # images = []
        generated_embeds = []
        generated_texts = []
        llava_gpts = []
        if "revised_generated_text" in samples[0]["json"]:
            revised_generated_texts = []
        else:
            revised_generated_texts = None
        
        if "gpt" in samples[0]["json"]:
            llava_gpts = []
        else:
            llava_gpts = None

        if self.build_info["use_input_embed"]:
            input_embed = []
        if self.build_info["use_output_embed"]:
            output_embed = []
        if (not self.build_info["use_input_embed"]) and (not self.build_info["use_output_embed"]):
            raise ValueError("No input or output embeds are used.")
        # input_token_ids = []
        output_token_ids = []

        input_embed_key = [key for key in samples[0].keys() if "input_embed" in key][0]
        output_embed_key = [key for key in samples[0].keys() if "output_embed" in key][0]
        for sample in samples:
            # images.append([sample["jpg"]])
            json_file = sample["json"]
            generated_texts.append(json_file["generated_text"])
            if llava_gpts is not None:
                llava_gpts.append(json_file["gpt"])
            # generated_embeds.append(sample["pth"])
            if revised_generated_texts is not None:
                revised_generated_texts.append(json_file["revised_generated_text"])
            
            if self.build_info["use_input_embed"]:
                input_embed.append(sample[input_embed_key])
            if self.build_info["use_output_embed"]:
                output_embed.append(sample[output_embed_key])

            # output_token_ids.append(torch.tensor(json_file["output_token_ids"]).long())
            output_token_ids.append(json_file["output_token_ids"])

        if self.build_info["use_input_embed"]:
            input_embed_max_len = self.build_info["input_embed_max_len"]
            max_input_embed_len = max([input_embed_i.shape[0] for input_embed_i in input_embed])
            input_embed_max_len = min(input_embed_max_len, max_input_embed_len)

            input_embed_mask = []
            for i in range(len(input_embed)):
                input_embed_i = input_embed[i]
                input_embed_len = input_embed_i.shape[0]
                if input_embed_len > input_embed_max_len:
                    input_embed_i = input_embed_i[:input_embed_max_len]
                    input_embed_mask_i = torch.ones(input_embed_max_len, device=input_embed_i.device, dtype=torch.long)
                elif input_embed_len < input_embed_max_len:
                    input_embed_i = F.pad(input_embed_i, (0, 0, 0, input_embed_max_len - input_embed_len))
                    input_embed_mask_i = torch.ones(input_embed_max_len, device=input_embed_i.device, dtype=torch.long)
                    input_embed_mask_i[input_embed_len:] = 0
                else:
                    input_embed_mask_i = torch.ones(input_embed_max_len, device=input_embed_i.device, dtype=torch.long)
                input_embed[i] = input_embed_i
                input_embed_mask.append(input_embed_mask_i)
            input_embed = torch.stack(input_embed, dim=0)
            input_embed_mask = torch.stack(input_embed_mask, dim=0)

        if self.build_info["use_output_embed"]:
            if self.build_info["random_split_output_embed"]:
                device = output_embed[0].device
                batch_size = len(output_embed)

                output_embed_parts = []
                output_embed_parts_masks = []
                output_token_ids_part_2 = []
                max_split_point = 0
                for i in range(batch_size):
                    output_embed_i = output_embed[i]
                    output_embed_len = output_embed_i.shape[0]
                    # randomly select a point
                    split_point = random.randint(1, min(output_embed_len - 1, self.build_info["output_embed_max_split_len"]))
                    max_split_point = max(max_split_point, split_point)
                    output_embed_i_part_1 = output_embed_i[:split_point]
                    output_embed_parts.append(output_embed_i_part_1)
                    output_embed_i_part_1_mask = torch.ones(output_embed_i_part_1.shape[0], device=device, dtype=torch.long)
                    output_embed_parts_masks.append(output_embed_i_part_1_mask)
                    output_token_ids_i_part_2 = output_token_ids[i][split_point:]
                    output_token_ids_part_2.append(output_token_ids_i_part_2)

                # pad and stack
                for i in range(batch_size):
                    output_embed_i = output_embed_parts[i]
                    output_embed_parts[i] = F.pad(output_embed_i, (0, 0, 0, max_split_point - output_embed_i.shape[0]))
                    output_embed_parts_masks[i] = F.pad(output_embed_parts_masks[i], (0, max_split_point - output_embed_i.shape[0]))
                
                output_embed = torch.stack(output_embed_parts, dim=0)
                output_embed_mask = torch.stack(output_embed_parts_masks, dim=0)
                output_token_ids = output_token_ids_part_2
            else:
                output_embed_max_len = self.build_info["output_embed_max_len"]
                max_output_embed_len = max([output_embed_i.shape[0] for output_embed_i in output_embed])
                output_embed_max_len = min(output_embed_max_len, max_output_embed_len)


                output_embed_list = []
                output_embed_mask_list = []
                output_token_ids_list = []

                for i in range(len(output_embed)):
                    output_embed_i = output_embed[i]
                    output_embed_len = output_embed_i.shape[0]
                    output_token_ids_i = output_token_ids[i]
                    if output_embed_len > output_embed_max_len:
                        output_embed_i = output_embed_i[:output_embed_max_len]
                        output_embed_mask_i = torch.ones(output_embed_max_len, device=output_embed_i.device, dtype=torch.long)
                        output_token_ids_i = output_token_ids_i[:output_embed_max_len]
                    elif output_embed_len < output_embed_max_len:
                        output_embed_i = F.pad(output_embed_i, (0, 0, 0, output_embed_max_len - output_embed_len))
                        output_embed_mask_i = torch.ones(output_embed_max_len, device=output_embed_i.device, dtype=torch.long)
                        output_embed_mask_i[output_embed_len:] = 0
                    else:
                        output_embed_mask_i = torch.ones(output_embed_max_len, device=output_embed_i.device, dtype=torch.long)
                    output_embed_list.append(output_embed_i)
                    output_embed_mask_list.append(output_embed_mask_i)
                    output_token_ids_list.append(output_token_ids_i)

                output_embed = torch.stack(output_embed_list, dim=0)
                output_embed_mask = torch.stack(output_embed_mask_list, dim=0)
                output_token_ids = output_token_ids_list

        input_embed_key = input_embed_key.replace(".pth", "")
        output_embed_key = output_embed_key.replace(".pth", "")
        
        inputs = {
            "generated_texts": generated_texts,
            # input_embed_key: input_embed,
            # output_embed_key: output_embed,
            "output_token_ids": output_token_ids,
            # "llava_gpts": llava_gpts
        }
        if llava_gpts is not None:
            inputs["llava_gpts"] = llava_gpts
        if revised_generated_texts is not None:
            inputs["revised_generated_texts"] = revised_generated_texts
        if self.build_info["use_input_embed"]:
            inputs[input_embed_key] = input_embed
            inputs["input_embed_mask"] = input_embed_mask
        if self.build_info["use_output_embed"]:
            inputs[output_embed_key] = output_embed
            inputs["output_embed_mask"] = output_embed_mask

        return inputs
