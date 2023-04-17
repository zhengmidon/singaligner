import random
import torch
from torch.utils.tensorboard import SummaryWriter
from .plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from .plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, alignments, log_prob, iteration, input_lengths, target_lengths):
        self.add_scalar("validation.loss", reduced_loss, iteration)

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx][:input_lengths[idx], :target_lengths[idx]].data.cpu().numpy(),
                info = "Phoneme"),
            iteration, dataformats='HWC')
        self.add_image(
            "ctc",
            plot_alignment_to_numpy(log_prob[idx][:input_lengths[idx], :].data.cpu().numpy(),
                info = "Alphabet"),
            iteration, dataformats='HWC')
