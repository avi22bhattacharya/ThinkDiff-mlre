import webdataset as wds
import tarfile
import os
import json
import tqdm
import subprocess
import multiprocessing
from functools import partial

def count_files_in_tar_3(tar_path):
    command = f"tar -tvf {tar_path} | wc -l"
    result = os.popen(command).read().strip()
    return int(result)

def process_tar_file(tar_path, item_number):
    tar_file_count = count_files_in_tar_3(tar_path)
    assert tar_file_count % item_number == 0
    tar_item_count = tar_file_count // item_number
    return {
        "url": tar_path,
        "nsamples": tar_item_count
    }


# storage = "example_cc_sbu_data/{00000..00002}.tar"
storage = "/project/visgroup/zmiaa/datasets/minigpt4/cc_sbu/cc_sbu_dataset/{00000..00002}.tar"
item_number = 3
output_dir = "minigpt4/configs/datasets/cc_sbu_mllama_vllm_process_wids"
shard_name = "wids_shards.json"

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, shard_name)
shard_list = wds.SimpleShardList(storage)
# shard_list["urls"]

output_json = {}
output_json["__kind__"] = "wids-shard-index-v1"
output_json["wids_version"] = 1
output_json["name"] = "LLaVA-Instruct-150K"
shardlist = []

process_tar_file_partial = partial(process_tar_file, item_number=item_number)
with multiprocessing.Pool(processes=4) as pool:
    shardlist = list(tqdm.tqdm(
        pool.imap(process_tar_file_partial, shard_list.urls),
        total=len(shard_list.urls)
    ))

output_json["shardlist"] = shardlist

json.dump(output_json, open(output_path, "w"), indent=4)