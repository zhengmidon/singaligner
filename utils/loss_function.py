from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from utils.utils import get_mask_from_lengths
from numba import jit, cuda
#from scipy.special import logsumexp
import time


class MelLoss(nn.Module):
    def __init__(self):
        super(MelLoss, self).__init__()

    def forward(self, mel_output, mel_gt, mel_lengths):
        mel_output = mel_output.permute(1, 2, 0) # [B,MB,N]
        tgt_mask = get_mask_from_lengths(mel_lengths).unsqueeze(1)
        mel_gt.requires_grad = False

        mel_loss = nn.MSELoss()(mel_output * tgt_mask, mel_gt)
        return mel_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, label_smooth=None, class_num=None):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output [B,vocab_size,T]
            target: ground truth of sampler [B,T]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing, in word scale
            B,vocab_size,T = pred.shape
            mask = target > 0
            mask = mask.unsqueeze(2).expand(B, T, vocab_size)
            mask = mask.to(pred.device)

            logprobs = F.log_softmax(pred, dim = 1)   # softmax + log
            logprobs = logprobs.transpose(1, 2)
            target = F.one_hot(target, self.class_num)  # 转换成one-hot,[B,T,vocab_size]
            
            target = torch.clamp(target.float(), min = self.label_smooth / (self.class_num - 1), max = 1.0 - self.label_smooth) # float必需
            loss = -1 * torch.sum(target * logprobs * mask, dim = 2)
        
        else:
            raise ModuleNotFoundError

        return loss.mean()

class GuidedAttentionLoss(nn.Module):
    def __init__(self, guide_sigma):
        super(GuidedAttentionLoss, self).__init__()
        self.guide_sigma = guide_sigma

    def forward(self, alignments, input_lengths, target_lengths):
        alignments = alignments.transpose(1, 2) # [B,T,N]
        guide_matrics = torch.zeros_like(alignments, device = alignments.device)
        for idx, alignment in enumerate(alignments):
            N, T = input_lengths[idx].cpu(), target_lengths[idx].cpu()
            weights_vec_left = [1 - np.exp(-(i /  N - 1.) ** 2 / (2 * self.guide_sigma ** 2)) for i in range(N)]
            weights_vec_right = [1 - np.exp(-(i /  N ) ** 2 / (2 * self.guide_sigma ** 2)) for i in range(N)]
            weights_vec = weights_vec_left + weights_vec_right
            offsets = [int((N / T) * i) for i in range(T)]
            guide_matric = []
            for i, offset in enumerate(offsets):
                row = weights_vec[N - offset:2 * N - offset]
                guide_matric.append(row)
            guide_matric = torch.tensor(guide_matric, device = alignments.device)
            guide_matrics[idx, :T, :N] = guide_matric

        score = guide_matrics * alignments
        guided_loss = score.mean()
        return guided_loss

class AntiBlankLoss(nn.Module):
    def __init__(self, blank_index):
        super(AntiBlankLoss, self).__init__()
        self.blank_index = blank_index

    def forward(self, log_softmax, input_lengths):
        softmax = log_softmax.exp()
        return softmax[:, :, self.blank_index].sum() / input_lengths.sum()

class TokenPredictLoss(nn.Module):
    def __init__(self):
        super(TokenPredictLoss, self).__init__()

    def forward(self, log_softmax, text_padded):
        log_softmax = log_softmax.transpose(0, 1)
        text_idxs = text_padded.unsqueeze(1).repeat(1, log_softmax.size(1), 1)
        gain = log_softmax.gather(dim = 2, index = text_idxs) # [B,N,T]

        token_mask = text_idxs > 0
        return -(gain * token_mask).sum() / token_mask.sum()


class ConnectionistAlignmentLoss(nn.Module):
    def __init__(self):
        super(ConnectionistAlignmentLoss, self).__init__()

    def logsumexp(self, _a, _b):
        '''
        torch.log(torch.exp(a) + torch.exp(b))
        '''
        a = torch.where(_a > _b, _a, _b)
        b = torch.where(_a < _b, _a, _b)
        return a + torch.log(1 + torch.exp(b - a))

    def compute_loss(self, cum_score_mat, score_mat, target_list):
        n, t = cum_score_mat.shape 
        cum_score_mat[:, 0] = torch.cumsum(score_mat[:, target_list[0]], dim=0)
        for i in range(1, n):
            pre_frame = cum_score_mat[i - 1]
            score_frame = score_mat[i, target_list[1:i + 1]]
            if i < t :
                prefix_score = self.logsumexp(pre_frame[0:i], pre_frame[1:i + 1])
            else:
                prefix_score = self.logsumexp(pre_frame[0:-1], pre_frame[1:])
            cum_score_mat[i, 1:i + 1] = score_frame + prefix_score
        return -cum_score_mat[-1, -1]

    def forward(self, scores, targets, input_lengths, target_lengths):
        """
        scores: log softmaxed logits, shape of [N, B, vocab_size]
        targets: alignment targets, shape of [B, T]
        input_lengths: lengths of scores, shape of [B]
        target_lengths: lengths of targets, shape of [B]

        """
        scores = scores.transpose(0, 1)
        b = targets.size(0)
        loss = torch.tensor(0., device=scores.device, dtype=torch.float, requires_grad=True)
        for idx, score_mat in enumerate(scores):
            score_mat = score_mat[:input_lengths[idx]]
            target_list = targets[idx, :target_lengths[idx]]
            n, v = score_mat.shape 
            t = target_list.shape[0]
            cum_score_mat = torch.ones(size=(n, t), dtype=torch.float, device=scores.device) * (-torch.inf)
            loss = loss + self.compute_loss(cum_score_mat, score_mat, target_list)
        return loss / b

class CALoss(nn.Module):
    def __init__(self, blank_index, blank_logprob=-1e6):
        super(CALoss, self).__init__()
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(blank=blank_index, zero_infinity=True)

    def forward(self, attn_logprob, target_seq, input_lengths, target_lengths):
        """
        attn_logprob: [N, B, vocab_size]
        """
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(0, 1, 0, 0, 0, 0),
                                value=self.blank_logprob)
        
        cost = self.CTCLoss(attn_logprob_pd,
                            target_seq,
                            input_lengths=input_lengths,
                            target_lengths=target_lengths)
        return cost

if __name__ == '__main__':
    bz = 16
    n = 600
    t = 60
    v = 60
    torch.manual_seed(42)
    _scores = torch.randn(n, bz, v, requires_grad=True, device="cuda:0")
    scores = _scores.log_softmax(2)
    targets = torch.randint(0, v, (bz, t), device="cuda:0")
    input_lengths = torch.tensor([n] * bz, device="cuda:0")
    target_lengths = torch.tensor([t] * bz, device="cuda:0")

    calossfn = ConnectionistAlignmentLoss()
    ctc_loss = torch.nn.CTCLoss()

    # start = time.time()
    # res = ctc_loss(scores, targets, input_lengths, target_lengths)
    # res.backward()
    # end = time.time()
    # print(f"ctcloss {res} grad {_scores.grad} cost time {end - start}")

    start = time.time()
    caloss = calossfn(scores, targets, input_lengths, target_lengths)
    mid = time.time()
    caloss.backward()
    end = time.time()
    print(f"caloss {caloss} grad {_scores.grad} forward time {mid - start}, backward time {end - mid}")





