import numpy as np
from scipy.io.wavfile import read
import torch
import librosa
import glob


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool() # pad之处为0
    return mask # [B,T]

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def load_wav_to_torch(full_path, sr):
    data, sampling_rate = librosa.load(full_path, sr = sr)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate, data


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def opencpop_load_filepaths_and_text(transcription_path, data_dir):
    with open(transcription_path, 'r', encoding='utf-8') as f:
        items = f.readlines()
    info_list =[]
    for item in items:
        info = item.strip().split('|')
        wav_path = f'{data_dir}/wavs/{info[0]}.wav'
        info_list.append([wav_path, info[2], info[5]])
    # t_len = lambda x : len(x[1].split(' '))
    # info_list = sorted(info_list, key = t_len)
    print(f"Dataset {transcription_path[:-4]} total {len(info_list)} items")
    return info_list

def opera_load_filepaths_and_text(prefix_file, data_dir):
    with open(prefix_file, 'r', encoding='utf-8') as f:
        items = f.readlines()
    info_list = []
    for item in items:
        item = item.strip()
        wav_paths = glob.glob(f'{data_dir}/opera_mfa/{item}/*.wav')
        for wav_path in wav_paths:
            text_path = wav_path.replace("wav", 'pho')
            with open(text_path, encoding = 'utf-8') as f:
                phones = f.readlines()
            phone_list = []
            gt_time_list = []
            for phone_item in phones:
                segs = phone_item.strip().split('\t')
                phone_list.append(segs[-1])
                gt_time_list.append(round(float(segs[1]) - float(segs[0]), 4))
            gt_time_list = [str(i) for i in gt_time_list]
            gt_time = ' '.join(gt_time_list)
            phone_list = ' '.join(phone_list)
            info_list.append([wav_path, phone_list, gt_time])
    print(f"Dataset {prefix_file[:-11]} total {len(info_list)} items")
    return info_list

def TIMIT_load_filepaths_and_text(prefix, data_dir, sampling_rate):
    wav_paths = glob.glob(f'{data_dir}/{prefix}/*/*/*.wav')
    info_list = []
    for wav_path in wav_paths:
        text_path = wav_path.replace("WAV.wav", 'PHN')
        with open(text_path, 'r', encoding = 'utf-8') as f:
            try:
                phones = f.readlines()
            except:
                print(text_path)
        phone_list = []
        gt_time_list = []
        for phone_item in phones:
            segs = phone_item.strip().split(' ')
            phone_list.append(segs[-1])
            gt_time_list.append(round((float(segs[1]) - float(segs[0])) / sampling_rate, 5))
        gt_time_list = [str(i) for i in gt_time_list]
        gt_time = ' '.join(gt_time_list)
        phone_list = ' '.join(phone_list)
        info_list.append([wav_path, phone_list, gt_time])
    print(f"Dataset {prefix} total {len(info_list)} items")
    return info_list

def NUS48E_load_filepaths_and_text(prefix, data_dir):
    wav_paths = glob.glob(f'{data_dir}/{prefix}/*/*.wav')
    info_list = []
    for wav_path in wav_paths:
        text_path = wav_path.replace("wav", 'txt')
        with open(text_path, 'r', encoding = 'utf-8') as f:
            try:
                phones = f.readlines()
            except:
                print(text_path)
        phone_list = []
        gt_time_list = []
        for phone_item in phones:
            segs = phone_item.strip().split(' ')
            phone_list.append(segs[-1])
            gt_time_list.append(round((float(segs[1]) - float(segs[0])), 5))
        gt_time_list = [str(i) for i in gt_time_list]
        gt_time = ' '.join(gt_time_list)
        phns = ' '.join(phone_list)
        info_list.append([wav_path, phns, gt_time])
    print(f"Dataset {prefix} total {len(info_list)} items")
    return info_list

def GeZiXi_load_filepaths_and_text(prefix, data_dir):
    wav_paths = glob.glob(f'{data_dir}/{prefix}/*.wav')
    info_list = []
    for wav_path in wav_paths:
        info_path = wav_path.replace("wav", 'info')
        with open(info_path, 'r', encoding = 'utf-8') as f:
            try:
                infos = f.readlines()
            except:
                print(info_path)
        phones = infos[0].strip()
        gt_times = infos[1].strip()
        info_list.append([wav_path, phones, gt_times])
    print(f"Dataset {prefix} total {len(info_list)} items")
    return info_list


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
