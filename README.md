# singaligner
A compact audio-to-phoneme aligner for singing voice.

The available datasets in our experiments are: [Opencpop](https://wenet.org.cn/opencpop/), [NamineRitsu](https://www.youtube.com/watch?v=pKeo9IE_L1I). One can experiment on your own datasets. 

Once the data is prepared, you should just do:
1. Create a virtual environment.
2. Define the dataloder and the collate function in utils/data_utils.py. You can inherit the existing classes.
3. Import your dataloader to train.py and change trainset, valset and collate_fn in prepare_dataloaders function.
4. Prepare a file named phone_set.json which contains the phone set of your dataset and put it at root of data_dir.
5. Change the data_dir to your data path in hparams.py
6. Run this command to start training: 
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --output_directory experiments/exp_name/ --log_directory tensorboard_logs
```
7. Run this command to start inferring:
```sh
CUDA_VISIBLE_DEVICES=0 python infer_prob.py --checkpoint_path experiments/exp_name/checkpoint_name \
--output_dir experiments/exp_name/
```
## Citation
```sh
@inproceedings{zheng2023compact,
  title={A Compact Phoneme-To-Audio Aligner for Singing Voice},
  author={Zheng, Meizhen and Bai, Peng and Shi, Xiaodong},
  booktitle={International Conference on Advanced Data Mining and Applications},
  pages={183--197},
  year={2023},
  organization={Springer}
}
```
