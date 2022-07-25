import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.common import get_object
from configs import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    cfg = Config(args.config, args.resume, args.reset)

    model = get_object(
        cfg("model", "name"),
        cfg("model", "kwargs"),
    )
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    transform_train = get_object(
        cfg("data", "transform", "train", "name"),
        cfg("data", "transform", "train", "kwargs"),
    )
    transform_val = get_object(
        cfg("data", "transform", "val", "name"),
        cfg("data", "transform", "val", "kwargs"),
    )
    dataset_train = get_object(
        cfg("data", "dataset", "train", "name"),
        {**cfg("data", "dataset", "train", "kwargs"), "transform": transform_train},
    )
    dataset_val = get_object(
        cfg("data", "dataset", "val", "name"),
        {**cfg("data", "dataset", "val", "kwargs"), "transform": transform_val},
    )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        loader_train = DataLoader(
            dataset_train,
            **cfg("data", "loader", "train", "kwargs"),
            sampler=sampler_train,
        )
        loader_val = DataLoader(
            dataset_val,
            **cfg("data", "loader", "val", "kwargs"),
            sampler=sampler_val,
        )
    else:
        loader_train = DataLoader(
            dataset_train,
            **cfg("data", "loader", "train", "kwargs"),
            shuffle=True,
        )
        loader_val = DataLoader(
            dataset_val,
            **cfg("data", "loader", "val", "kwargs"),
            shuffle=False,
        )

    optimizer = get_object(
        cfg("optimizer", "name"),
        {"params": model.parameters(), **cfg("optimizer", "kwargs")},
    )
    logger = get_object(
        cfg("logger", "name"),
        {"logdir": cfg.logdir, "resume": args.resume, **cfg("logger", "kwargs")},
    )
    trainer = get_object(
        cfg("trainer", "name"),
        {
            "model": model,
            "optimizer": optimizer,
            "loader_train": loader_train,
            "loader_val": loader_val,
            "logger": logger,
            "root": cfg.root,
            "distributed": args.distributed,
            **cfg("trainer", "kwargs"),
        },
    )

    trainer.run(args.resume)


if __name__ == "__main__":
    main()
