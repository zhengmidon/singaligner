import os
import time
import argparse
import math
from numpy import finfo
import random  
import numpy as np

import torch
import torch.nn as nn
from utils.distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader as DataLoader

# from model import ATTNAligner
from model_comp import ATTNAligner
# from model_conformer import ATTNAligner
# from model_conv import ATTNAligner
# from model_linear import ATTNAligner
from model_transformer import ATTNAlignerT
from utils.data_utils import OPENCPOPTextMelLoader, OPENCPOPTextMelCollate, OperaTextMelLoader, OperaTextMelCollate
from utils.data_utils import TIMITTextMelLoader, TIMITTextMelCollate, NUS48ETextMelLoader, NUS48ETextMelCollate
from utils.data_utils import GeZiXiTextMelLoader, GeZiXiTextMelCollate, NamineRitsuTextMelLoader, NamineRitsuTextMelCollate
from utils.data_utils import build_phone_encoder
from utils.loss_function import MelLoss, LabelSmoothingCrossEntropy, GuidedAttentionLoss, CALoss
from utils.logger import Tacotron2Logger
from hparams import hparams
from hparams_transformer import hparams as hparams_t
from utils.DPAlign import DurationExtraction, CTCAlignment


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


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready

    # trainset = OPENCPOPTextMelLoader(hparams.training_transcription, hparams, hparams.data_dir)
    # valset = OPENCPOPTextMelLoader(hparams.validation_transcription, hparams, hparams.data_dir)
    # collate_fn = OPENCPOPTextMelCollate(hparams.data_dir)

    # trainset = TIMITTextMelLoader('TRAIN', hparams, hparams.data_dir)
    # valset = TIMITTextMelLoader('TEST', hparams, hparams.data_dir)
    # collate_fn = TIMITTextMelCollate(hparams.data_dir)

    # trainset = OperaTextMelLoader(hparams.train_prefix, hparams, hparams.data_dir)
    # valset = OperaTextMelLoader(hparams.validate_prefix, hparams, hparams.data_dir)
    # collate_fn = OperaTextMelCollate(hparams.data_dir)

    # trainset = NUS48ETextMelLoader('train', hparams, hparams.data_dir)
    # valset = NUS48ETextMelLoader('test', hparams, hparams.data_dir)
    # collate_fn = NUS48ETextMelCollate(hparams.data_dir)

    trainset = GeZiXiTextMelLoader('train', hparams, hparams.data_dir)
    valset = GeZiXiTextMelLoader('test', hparams, hparams.data_dir)
    collate_fn = GeZiXiTextMelCollate(hparams.data_dir)

    # trainset = NamineRitsuTextMelLoader('train', hparams, hparams.data_dir)
    # valset = NamineRitsuTextMelLoader('test', hparams, hparams.data_dir)
    # collate_fn = NamineRitsuTextMelCollate(hparams.data_dir)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=hparams.num_workers, shuffle=True,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = ATTNAligner(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

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


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def print_model_info(model):
    print("Model Arch:\n")
    print(model)
    param_num = 0
    for name,parameters in model.named_parameters():
        param_num = param_num + parameters.numel()
    print(f"\nParameters:{param_num / 1e6:.2f} M")

def set_seed(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def validate(model, criterions, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, blank_index, training_time_start):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=4,
                                shuffle=True, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        total_iters = 0
        total_time = 0.0
        max_error = 0.0
        min_error = 10.0
        over_lap = 0.0
        total_dur = 0.0
        total_resluts = []
        for i, batch in enumerate(val_loader):
            x, y, text, split_gt = model.parse_batch(batch)
            text_padded, text_lengths, mel_padded, mel_lengths = x

            mel_output, log_prob, alignments_output = model(x)
            ctc_loss = criterions[0](log_prob, y, input_lengths = x[3], target_lengths = x[1])
            mel_loss = hparams.mel_loss_weight * criterions[1](mel_output, mel_gt = x[2], mel_lengths = x[3])
            loss = ctc_loss + mel_loss
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss

            #log_prob = log_prob.transpose(0, 1)
            log_prob = log_prob.cpu() # [N,B,vocab_size]
            log_prob = log_prob.exp().transpose(0, 1)
            text_idxs = text_padded.unsqueeze(1).repeat(1, log_prob.size(1), 1).cpu()
            alignments = log_prob.gather(dim = 2, index = text_idxs) # [B,N,T]
            # # alignments = alignments + log_prob.gather(dim = 2, index=alignments.new_full(alignments.shape, blank_index).long())

            for idx, alignment in enumerate(alignments):
                # log_p = log_p[:mel_lengths[idx], :]
                # target = text_padded[idx, :text_lengths[idx]]
                alignment = alignment[:mel_lengths[idx], :text_lengths[idx]]
                # alignment = CTCAlignment(log_p, target, blank_index)
                # alignment = torch.from_numpy(alignment)
                duration = DurationExtraction(alignment.T)
                duration = [i * hparams.hop_length / hparams.sampling_rate for i in duration]
                duration_str = [str(round(i, 5)) for i in duration]
                

                gt_times = split_gt[idx].strip().split(' ')
                gt_times = [float(i) for i in gt_times]
                assert len(gt_times) == len(duration)
                total_iters += len(duration)
                for ind, t in enumerate(duration):
                    res = abs(t - gt_times[ind])
                    total_resluts.append(res)
                    total_time += res
                    if res > max_error:
                        max_error = res
                    if res < min_error:
                        min_error = res

                cum_pd_time = np.cumsum(duration)
                cum_gt_time = np.cumsum(gt_times)
                total_dur += cum_gt_time[-1]
                pd_start = 0.0 
                gt_start = 0.0
                for idx, gt_t in enumerate(cum_gt_time):
                    pd_t = cum_pd_time[idx]
                    if pd_t <= gt_start or gt_t <= pd_start:
                        pd_start = pd_t
                        gt_start = gt_t
                        continue
                    elif gt_t >= pd_t:
                        if gt_start >= pd_start:
                            over_lap += pd_t - gt_start
                            pd_start = pd_t
                            gt_start = gt_t
                        else:
                            over_lap += pd_t - pd_start
                            pd_start = pd_t
                            gt_start = gt_t
                    else:
                        if gt_start >= pd_start:
                            over_lap += gt_t - gt_start
                            pd_start = pd_t
                            gt_start = gt_t
                        else:
                            over_lap += gt_t - pd_start
                            pd_start = pd_t
                            gt_start = gt_t

        val_loss = val_loss / (i + 1)
        ade = round(total_time / total_iters, 5)
        
    model.train()
    total_resluts.sort()
    if rank == 0:
        print("Validation loss {}: {:6f}  ".format(iteration, val_loss))
        print(f"===MAE:{ade}s, ===MED {total_resluts[len(total_resluts)//2]}s, ===PCAS {over_lap / total_dur:.4f}, \
            Min Error {min_error:.6f} s, Max Error {max_error:.6f} s, Training time {time.time() - training_time_start}s")
        logger.log_validation(val_loss, model, alignments, log_prob, iteration,\
                                 input_lengths = x[3], target_lengths = x[1])
    return round(val_loss, 3), ade


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
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
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    set_seed(hparams.seed)
    _text_encoder = build_phone_encoder(hparams.data_dir)
    #hparams.blank_index = _text_encoder.vocab_size

    model = load_model(hparams)
    print_model_info(model)

    learning_rate = hparams.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = hparams.lr_scheduler_gamma)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    ctc_loss_fn = nn.CTCLoss(blank = hparams.blank_index) # blank implies the sil index
    mel_loss_fn = MelLoss()
    #galoss_fn = GuidedAttentionLoss(guide_sigma = hparams.guide_sigma)

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    loss_chk_cycle = 0.
    ctc_loss_chk_cycle = 0.
    mel_loss_chk_cycle = 0.
    #ga_loss_chk_cycle = 0.

    min_val_loss = 100.
    min_ade = 100.
    start = time.perf_counter()
    tolerance = hparams.tolerance
    tol_count = 0
    check_cycle = hparams.check_cycle
    blank_total = 0
    frame_total = 0
    training_time_start = time.time()
    for epoch in range(epoch_offset, hparams.epochs):
        if epoch == 1:
            print("Epoch: {}, {} iters per epoch".format(epoch, iteration))
        for i, batch in enumerate(train_loader):
            time.sleep(0.003)
            model.zero_grad()
            x, y, text, split_gt = model.parse_batch(batch)
            # if epoch == 0 and i == 0:
            #     print(f"Data Example:\n \
            #         text:{text}, split_gt:{split_gt}, \
            #         text_padded:{x[0]}, text_lengths:{x[1]}, \
            #         mel_padded:{x[2]},mel_shape:{x[2].shape}, \
            #         mel_lengths:{x[3]}")
            mel_output, log_prob, alignments = model(x) # log_prob : [N,B,vocab_size]
            blank_total += (log_prob.cpu().argmax(dim=2) == hparams.blank_index).sum().item()
            frame_total += x[3].cpu().sum().item()

            ctc_loss = ctc_loss_fn(log_prob, y, input_lengths = x[3], target_lengths = x[1])
            mel_loss = hparams.mel_loss_weight * mel_loss_fn(mel_output, mel_gt = x[2], mel_lengths = x[3])
            #ga_loss = hparams.ga_loss_weight * galoss_fn(alignments, input_lengths = x[3], target_lengths = x[1])

            loss = mel_loss + ctc_loss

            loss_chk_cycle += loss.cpu().item()
            ctc_loss_chk_cycle += ctc_loss.cpu().item()
            mel_loss_chk_cycle += mel_loss.cpu().item()
            #ga_loss_chk_cycle += ga_loss.cpu().item()

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and (iteration + 1) % check_cycle == 0:
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                duration = time.perf_counter() - start
                print("Iter {} | Train loss {:.6f} mel_loss {:.6f} ctc_loss {:.6f} \
                    lr {:.6f} blank ratio {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, loss_chk_cycle / check_cycle, \
                    mel_loss_chk_cycle / check_cycle, ctc_loss_chk_cycle / check_cycle, \
                    current_lr, blank_total / frame_total, grad_norm, duration / check_cycle))
                logger.log_training(
                    loss_chk_cycle / check_cycle, grad_norm, current_lr, duration, iteration)

                loss_chk_cycle = 0.
                ctc_loss_chk_cycle = 0.
                mel_loss_chk_cycle = 0.
                #ga_loss_chk_cycle = 0.
                blank_total = 0
                frame_total = 0
                start = time.perf_counter()

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                val_loss, ade = validate(model, [ctc_loss_fn, mel_loss_fn], valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, hparams.blank_index, training_time_start)
                # if val_loss < min_val_loss:
                #     min_val_loss = val_loss
                #     tol_count = 0
                # elif val_loss >= min_val_loss and tol_count < tolerance:
                #     tol_count += 1
                #     iteration += 1
                #     continue
                # else:
                #     print(f" Current validation loss {val_loss}, min_val_loss {min_val_loss}, exceed tolerance, early stopping!")
                #     return

                if ade < min_ade:
                    min_ade = ade
                    tol_count = 0
                elif ade >= min_ade and tol_count < tolerance:
                    tol_count += 1
                    iteration += 1
                    continue
                else:
                    print(f" Current Average Duration Error {ade}, min_ade {min_ade}, exceed tolerance, early stopping!")
                    return

                if rank == 0 and hparams.save_model:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}_{}".format(iteration, val_loss))
                    save_checkpoint(model, optimizer, current_lr, iteration,
                                    checkpoint_path)

            iteration += 1
        lr_scheduler.step()
    print("All Done！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    hparams.print_attrs()

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
