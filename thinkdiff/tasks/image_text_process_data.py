"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from thinkdiff.common.registry import registry
from thinkdiff.tasks.base_task import BaseTask
import torch
from thinkdiff.datasets.data_utils import prepare_sample
import webdataset as wds
import os
import tqdm
import io

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@registry.register_task("image_text_process_data")
class ImageTextProcessDataTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        amp_dtype=torch.float16,
        use_clip_grad_norm=False,
        max_grad_norm=1.0,
        output_shard_path=None
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        assert output_shard_path
        os.makedirs(output_shard_path[0], exist_ok=True)
        output_shard_path_1 = os.path.join(output_shard_path[0], output_shard_path[1])
        start_shard = output_shard_path[2]
        data_len = len(data_loader.dataset) // data_loader.batch_size
        with wds.ShardWriter(output_shard_path_1, maxsize=(10**8) * 5, start_shard=start_shard) as writer:  # Create writer
            # for samples in tqdm.tqdm(data_loader):
            progress_bar = tqdm.tqdm(total=data_len)
            for samples in data_loader:
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                batch_size = len(samples["images"])
                
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    output = model(samples)

                    if "generated_embed" in output:
                        generated_embed = output["generated_embed"]
                    else:
                        generated_embed = None
                    generated_text = output["generated_text"]
                    mllama_output_token = output["generated_token"]
                    if generated_embed is not None:
                        generated_embed_dict = flatten_dict(generated_embed)

                for i in range(batch_size):
                    json_txt = samples["jsons"][i]
                    json_txt["generated_text"] = generated_text[i]

                    json_txt["input_prompt"] = mllama_output_token["input_prompt"][i]
                    json_txt["input_prompt_token_ids"] = mllama_output_token["input_prompt_token_ids"][i]
                    json_txt["output_text"] = mllama_output_token["output_text"][i]
                    json_txt["output_token_ids"] = list(mllama_output_token["output_token_ids"][i])

                    key = samples["filenames"][i]
                    # embed = generated_embed[i].cpu()
                    # buffer = io.BytesIO()
                    # torch.save(embed.clone(), buffer)
                    write_dict = {}
                    write_dict["__key__"] = key
                    write_dict["jpg"] = samples["images"][i][0]
                    write_dict["json"] = json_txt
                    if generated_embed is not None:
                        for k, v in generated_embed_dict.items():
                            v_i = v[i].cpu()
                            buffer = io.BytesIO()
                            torch.save(v_i.clone(), buffer)  # Clone the tensor here
                            write_dict[f"{k}.pth"] = buffer.getvalue()
                    
                    writer.write(write_dict)
                progress_bar.update(1)

                    # writer.write({
                    #     "__key__": key,
                    #     "jpg": samples["images"][i][0],              # Save the image data
                    #     "json": json_txt,               # Save the text data
                    #     "pth": buffer.getvalue()    # Save the extra tensor data in .pth format
                    # })

                # for i in range(batch_size):
                #     json_txt = samples["jsons"][i]
                #     json_txt["generated_text"] = generated_text[i]
                #     key = samples["filenames"][i]
                #     embed = generated_embed[i].cpu()
                #     buffer = io.BytesIO()
                #     torch.save(embed.clone(), buffer)
                #     writer.write({
                #         "__key__": key,
                #         "jpg": samples["images"][i][0],              # Save the image data
                #         "json": json_txt,               # Save the text data
                #         "pth": buffer.getvalue()    # Save the extra tensor data in .pth format
                #     })
        # seen_files = set()
        # output_dup = []
        # cnt = 0
        # for samples in tqdm.tqdm(data_loader):
        #     batch_size = len(samples["images"])
        #     for i in range(batch_size):
        #         print("checking", cnt)
        #         cnt += 1
        #         base_name = samples["filenames"][i]
        #         if base_name in seen_files:
        #             output_dup.append(base_name)
        #             print(f"dup detected: {base_name}")
        #         else:
        #             seen_files.add(base_name)
        
        # print("output_dup", output_dup)

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        amp_dtype=torch.float16,
        use_clip_grad_norm=False,
        max_grad_norm=1.0,
        output_shard_path=None
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            amp_dtype=amp_dtype,
            use_clip_grad_norm=use_clip_grad_norm,
            max_grad_norm=max_grad_norm,
            output_shard_path=output_shard_path
        )