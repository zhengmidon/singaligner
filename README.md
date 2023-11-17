# singaligner
A compact audio-to-phoneme aligner for singing voice.

The available datasets in our experiments are: [Opencpop](https://wenet.org.cn/opencpop/), [NamineRitsu](https://www.youtube.com/watch?v=pKeo9IE_L1I). One can experiment on your own datasets. 

Once the data is prepared, you should just do:
1. Create a virtual environment.
2. Define the dataloder and the collate function in utils/data_utils.py. You can inherit the existing classes.
3. Change the data_dir to your data path in hparams.py
4. Run this command to start training: 
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --output_directory experiments/0/ --log_directory tensorboard_logs
```
## Citation
    @inproceedings{zheng2023compact,
  title={A Compact Phoneme-To-Audio Aligner for Singing Voice},
  author={Zheng, Meizhen and Bai, Peng and Shi, Xiaodong},
  booktitle={International Conference on Advanced Data Mining and Applications},
  pages={183--197},
  year={2023},
  organization={Springer}
}
