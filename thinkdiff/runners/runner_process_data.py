"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from thinkdiff.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from thinkdiff.common.registry import registry
from thinkdiff.common.utils import is_url
from thinkdiff.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from thinkdiff.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from thinkdiff.runners.runner_base import RunnerBase
import wids

@registry.register_runner("runner_process_data")
class RunnerProcessData(RunnerBase):
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)
    
    @property
    def model(self):
        return self._model

    @property
    def output_shard_path(self):
        output_shard_path = self.config.run_cfg.get("output_shard_path", None)
        return output_shard_path

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = DataLoader(
                        dataset,
                        batch_size=bsz[0],
                        num_workers=num_workers,
                        pin_memory=True,
                        collate_fn=collate_fn[0]
                    )
            elif isinstance(dataset, wids.ShardListDataset):
                # sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
                sampler = wids.ChunkedSampler(dataset, chunksize=1000, shuffle=True)
                loader = DataLoader(
                    dataset, batch_size=bsz[0], sampler=sampler, num_workers=num_workers, collate_fn=collate_fn[0]
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler

                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                print("only support one dataset")
                # if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                #     dataset_ratios = [d.sample_ratio for d in dataset]
                # loader = MultiIterLoader(
                #     loaders=[
                #         _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i])
                #         for i, d in enumerate(dataset)
                #     ],
                #     ratios=dataset_ratios,
                # )
                loader = _create_loader(dataset[0], num_workers, bsz, is_train, collate_fn)
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    def train(self):
        self.log_config()
        for cur_epoch in range(1):
            logging.info("Start training")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(split_name="train", stats=train_stats)

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
            amp_dtype=self.amp_dtype,
            use_clip_grad_norm=self.use_clip_grad_norm,
            max_grad_norm=self.max_grad_norm,
            output_shard_path=self.output_shard_path
        )
