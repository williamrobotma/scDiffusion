"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.cell_datasets_loader import load_data, get_dataset, dataset_to_loader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import torch
import torch.nn as nn
import numpy as np

def main():
    args = create_argparser().parse_args()

    setup_seed(1234)

    dist_util.setup_dist()
    logger.configure()

    if not args.device_ids:
        device_ids = [dist_util.dev()]
    else:
        device_ids = [th.device(device_id) for device_id in args.device_ids]
    logger.log("creating data loader...")
    dataset = get_dataset(
        data_dir=args.data_dir,
        vae_path=args.vae_path,
        train_vae=False,
        hidden_dim=args.latent_dim,
        train_split_only=args.train_split_only,
        device=device_ids[0],
    )
    num_class = np.unique(dataset.class_name).shape[0]
    data = dataset_to_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     vae_path=args.vae_path,
    #     hidden_dim=args.latent_dim,
    #     train_vae=False,
    # )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            vae_path=args.vae_path,
            hidden_dim=args.latent_dim,
            train_vae=False,
            num_workers=args.num_workers,
        )
    else:
        val_data = None

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys()),
        num_class=num_class,
    )
    device = device_ids[0]
    model.to(device)
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(args.resume_checkpoint, map_location=device)
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=device_ids,
        output_device=device,
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=True,
    )

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(dist_util.load_state_dict(opt_checkpoint, map_location=device))

    logger.log("training classifier model...")

    if args.noised:

        def noise_and_t(batch):
            t, _ = schedule_sampler.sample(
                batch.shape[0],
                device=device,
                start_guide_time=args.start_guide_time,
            )
            batch = diffusion.q_sample(batch, t)
            return batch, t

    else:

        def noise_and_t(batch):
            return batch, th.zeros(batch.shape[0], dtype=th.long, device=device)

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(device, non_blocking=True)

        batch = batch.to(device, non_blocking=True)
        # Noisy cells
        batch, t = noise_and_t(batch)

        mp_trainer.zero_grad()
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            with th.no_grad():
                losses = {}
                losses[f"{prefix}_loss"] = loss.detach()
                losses[f"{prefix}_acc@1"] = compute_top_k(logits, sub_labels, k=1, reduction="none")

                log_loss_dict(diffusion, sub_t, losses)
                del losses
            loss = loss.mean()
            if loss.requires_grad:
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    if args.anneal_lr:
        lr_setter = set_annealed_lr
    else:
        lr_setter = set_dummy_lr

    model_path = args.model_path
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        lr_setter(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step, model_path)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step, model_path)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def set_dummy_lr(opt, base_lr, frac_done):
    return


def save_model(mp_trainer, opt, step, model_path):
    if dist.get_rank() == 0:
        model_dir = model_path
        os.makedirs(model_dir,exist_ok=True)
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(model_dir, f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(model_dir, f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad",
        val_data_dir="",
        noised=True,
        iterations=500000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=128,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=100,
        eval_interval=100,
        save_interval=100000,
        vae_path="output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt",
        latent_dim=128,
        model_path="output/classifier_checkpoint/classifier_muris",
        start_guide_time=500,
        num_workers=1,
        # num_class=12,
    )
    # num_class = defaults['num_class']
    defaults.update(classifier_and_diffusion_defaults())
    # defaults['num_class']= num_class
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument("--train_split_only", action="store_true")
    parser.add_argument("--device_ids", nargs="*", default=None)
    return parser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
