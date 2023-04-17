import random
import numpy as np
import torch
import torch.utils.data
import os
import json

import sys,os
sys.path.append('..')

import module.layers as layers
from utils.utils import load_wav_to_torch, opencpop_load_filepaths_and_text, opera_load_filepaths_and_text, \
                        GeZiXi_load_filepaths_and_text
from utils.utils import NUS48E_load_filepaths_and_text, TIMIT_load_filepaths_and_text
from .text_encoder import TokenTextEncoder
from hparams import hparams as hp
import librosa
#from resemblyzer import VoiceEncoder, preprocess_wav


def build_phone_encoder(data_dir):
    phone_list_file = os.path.join(data_dir, 'phone_set.json')
    phone_list = json.load(open(phone_list_file))
    return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')

class OPENCPOPTextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, transcription_path, hparams, data_dir):
        self.audiopaths_and_text = opencpop_load_filepaths_and_text(transcription_path, data_dir)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.text_encoder = build_phone_encoder(data_dir)
        #self.voice_encoder = VoiceEncoder(device="cpu")

        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, t_gt = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        tokens = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (tokens, mel, text, t_gt)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate, origin_audio = load_wav_to_torch(filename, hp.sampling_rate)
            # processed_audio = preprocess_wav(origin_audio, source_sr = sampling_rate)
            # with torch.no_grad():
            #     utterance_embedding = self.voice_encoder.embed_utterance(processed_audio)

            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            #audio_norm = audio / self.max_wav_value
            audio_norm = audio.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(self.text_encoder.encode(text))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class OPENCPOPTextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, data_dir):
        self.text_encoder = build_phone_encoder(data_dir)

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        text_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = text_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()

        mel_lengths = torch.LongTensor(len(batch))
        text = []
        split_gt = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)
            text.append(batch[ids_sorted_decreasing[i]][2])
            split_gt.append(batch[ids_sorted_decreasing[i]][3])

        return text_padded, text_lengths, mel_padded, mel_lengths, text, split_gt

class OperaTextMelLoader(OPENCPOPTextMelLoader):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, prefix_file, hparams, data_dir):
        self.audiopaths_and_text = opera_load_filepaths_and_text(prefix_file, data_dir)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.text_encoder = build_phone_encoder(data_dir)
        #self.voice_encoder = VoiceEncoder(device="cpu")
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)



class OperaTextMelCollate(OPENCPOPTextMelCollate):
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, data_dir):
        self.text_encoder = build_phone_encoder(data_dir)


class TIMITTextMelLoader(OPENCPOPTextMelLoader):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, prefix, hparams, data_dir):
        self.audiopaths_and_text = TIMIT_load_filepaths_and_text(prefix, data_dir, hparams.sampling_rate)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.text_encoder = build_phone_encoder(data_dir)
        #self.voice_encoder = VoiceEncoder(device="cpu")
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)


class TIMITTextMelCollate(OPENCPOPTextMelCollate):
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, data_dir):
        self.text_encoder = build_phone_encoder(data_dir)


class NUS48ETextMelLoader(OPENCPOPTextMelLoader):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, prefix, hparams, data_dir):
        self.audiopaths_and_text = NUS48E_load_filepaths_and_text(prefix, data_dir)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.text_encoder = build_phone_encoder(data_dir)
        #self.voice_encoder = VoiceEncoder(device="cpu")
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)


class NUS48ETextMelCollate(OPENCPOPTextMelCollate):
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, data_dir):
        self.text_encoder = build_phone_encoder(data_dir)


class GeZiXiTextMelLoader(OPENCPOPTextMelLoader):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, prefix, hparams, data_dir):
        self.audiopaths_and_text = GeZiXi_load_filepaths_and_text(prefix, data_dir)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.text_encoder = build_phone_encoder(data_dir)
        #self.voice_encoder = VoiceEncoder(device="cpu")
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)


class GeZiXiTextMelCollate(OPENCPOPTextMelCollate):
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, data_dir):
        self.text_encoder = build_phone_encoder(data_dir)

class NamineRitsuTextMelLoader(GeZiXiTextMelLoader):
    pass


class NamineRitsuTextMelCollate(GeZiXiTextMelCollate):
    pass
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
