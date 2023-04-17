import os
import time
import argparse
import math
from numpy import finfo

import torch
from utils.distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import ATTNAligner
from model_transformer import ATTNAlignerT
from utils.data_utils import OPENCPOPTextMelLoader, OPENCPOPTextMelCollate
from utils.loss_function import MelLoss, LabelSmoothingCrossEntropy, GuidedAttentionLoss
from utils.logger import Tacotron2Logger
from utils.DPAlign import DurationExtraction
from hparams import hparams
from hparams_transformer import hparams as hparams_t
import numpy as np

torch.set_printoptions(threshold=50000)
np.set_printoptions(threshold=50000)


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)
    print("Done initializing distributed")

def prepare_test_dataloader(hparams):
    # Get data, data loaders and collate function ready
    testset = OPENCPOPTextMelLoader(hparams.validation_transcription, hparams, hparams.data_dir)
    collate_fn = OPENCPOPTextMelCollate(hparams.data_dir)

    if hparams.distributed_run:
        test_sampler = DistributedSampler(testset)
        shuffle = False
    else:
        test_sampler = None
        shuffle = True

    test_loader = DataLoader(testset, num_workers=hparams.num_workers, shuffle=False,
                              sampler=test_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)
    return test_loader, testset, collate_fn


def load_model(hparams):
    model = ATTNAligner(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    return model

def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, iteration

def infer(model, dataloader, output_dir, hparams):
    """Handles all the validation scoring and printing"""
    out_f = open(f"{output_dir}/alignments.txt", 'w', encoding = 'utf-8')
    model.eval()
    s_time = time.perf_counter()
    print('Start infering')
    total_iters = 0
    total_time = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y, text, split_gt = model.parse_batch(batch)
            _, text_lengths, _, mel_lengths = x
            _, _, alignments = model(x)
            alignments = alignments.cpu()

            for idx, alignment in enumerate(alignments):
                alignment = alignment[:mel_lengths[idx], :text_lengths[idx]]
                #print(alignment[:,-1])
                #print(f"len alignment {alignment.size(0)}, text {text[idx]} len text {len(text[idx].split(' '))}")
                duration = DurationExtraction(alignment.T)
                duration = [i * hparams.hop_length / hparams.sampling_rate for i in duration]
                duration_str = [str(round(i, 5)) for i in duration]
                out_f.write(f"Text|{text[idx]} \n")
                out_f.write(f"Split_gt|{split_gt[idx]} \n")
                out_f.write(f"Split_estimated|{' '.join(duration_str)} \n")

                gt_times = split_gt.strip().split(' ')
                gt_times = [float(i) for i in gt_times]
                assert len(gt_times) == len(duration)
                total_iters += len(duration)
                for idx, time in enumerate(duration):
                    res = abs(time - gt_times[idx])
                    total_time += res

    print(f"Average Duration Erros:{round(total_time / total_iters, 6)} s")
    out_f.write(f"\nAverage Duration Erros:{round(total_time / total_iters, 6)} s")


    print(f"Done, consume time {time.perf_counter() - s_time}")
    out_f.close()


def test(checkpoint_path, warm_start, output_dir, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)

    if hparams.fp16_run:
        from apex import amp
        model, = amp.initialize(
            model,  opt_level='O2')

    test_loader, testset, collate_fn = prepare_test_dataloader(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, iteration = load_checkpoint(
                checkpoint_path, model,)

    model.eval()
    infer(model, test_loader, output_dir, hparams)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=True, help='checkpoint path')
    parser.add_argument('--output_dir', type=str, default='experiments/',
                        required=True, help='Output save dir')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()

    test(args.checkpoint_path,
          args.warm_start, args.output_dir, hparams)
