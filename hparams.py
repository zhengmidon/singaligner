from glob import glob
import os

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value

    def print_attrs(self):
        for key in self.data.keys():
            print(f"{key}={self.data[key]}")


# Default hyperparameters
hparams = HParams(
    ################################
    # Experiment Parameters        #
    ################################
    epochs=500, # max epoches
    iters_per_checkpoint=100, # save checkpoint every this iters
    check_cycle=100, # validation period
    seed=42,
    num_workers=8,
    dynamic_loss_scaling=True,
    fp16_run=False,
    distributed_run=False,
    dist_backend="nccl",
    dist_url="tcp://localhost:54321",
    cudnn_enabled=True,
    cudnn_benchmark=False,
    ignore_layers=['embedding.weight'],

    ################################
    # Data Parameters             #
    ################################
    load_mel_from_disk=False,

    data_dir='DATA/GEZIXI_ALIGN',

    ################################
    # Audio Parameters             #
    ################################
    max_wav_value=32768.0,
    sampling_rate=24000,
    filter_length=512,
    hop_length=128,
    win_length=512,
    n_mel_channels=80,
    mel_fmin=10.0,
    mel_fmax=8000.0,

    ################################
    # Model Parameters             #
    ################################
    prenet_type='conv2d', # linear or conv1d or conv2d
    hidden_size=256,

    # MelEncoder parameters
    prenet_k_sizes_1d=[3,3],
    prenet_d_sizes_1d=[3,3],

    prenet_k_sizes_2d=[(20,3), (60,1), ],
    prenet_d_sizes_2d=[(1,1), (1,1), ],
    strides_2d=[(10,1), (10,1), ],
    pool_k_sizes_2d=[(7,3), (3,3), ],

    melencoder_layers=2,

    # TextEncoder parameters
    textencoder_layers=2,
    
    # Attention parameters
    num_heads=4,

    # CTCdecoder paramaters
    ctcdecoder_layers=2,

    # meldecoder parameters
    meldecoder_layers=2,

    ################################
    # Optimization Hyperparameters #
    ################################
    lr_scheduler_gamma=0.93,
    blank_index=2, # ctc blank index
    mel_loss_weight=1,
    use_saved_learning_rate=True,
    learning_rate=1e-3,
    weight_decay=1e-6,
    grad_clip_thresh=1.0,
    batch_size=16,
    tolerance=5,
    save_model=False,
    mask_padding=True  # set model's padded outputs to padded values
)


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)
