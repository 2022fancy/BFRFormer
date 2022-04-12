# BFRFormer



This repo is the official implementation of BFRFormer: Adversarial Consistency Transformer for Blind Face Restoration

By Anonymous Author(s)

## Abstract

> Abstract
 Due to the high illness and the complex unknown degradation process, blind face restoration (BFR) from severely degraded face images is a very challenging problem. The current mainstream approaches of blind face restoration are based on convolution neural network,  which mainly contain two stages because directly training a deep neural network usually cannot lead to acceptable results. Besides, these methods need to design complex strategies and various supervised loss functions.
 To overcome these drawbacks,this paper proposes a transformer-based blind face restoration baseline network named BFRFormer in which the training process is end-to-end. BFRFormer contains five parts: degradation module, encoder module, decoder module, mapping module and discriminator module. Specifically, the decoder module is composed of several StyleSwin Transformer blocks (STB), each of which has several StyleSwin layers as well as a semantic feature input extracted from the encoder module. Local attention is introduced to achieve a trade-off between computational complexity and modeling capacity. Wavelet discriminator is introduced to stablize the process of adversarial training and suppress the blocking artifacts. The proposed BFRFormer method is easy-to-implement and easy-to--train, and it can generate visually realistic and fidelity results. Sufficient experiments demonstrate that the proposed method achieves the SOTA performance compared to other blind face restoration methods based on CNN. 

## Quantitative Results



## Requirements

To install the dependencies:


```bash
python -m pip install -r requirements.txt
```

## Eval FID && LPIPS

To evaluate the fid score:

```bash
python -m torch.distributed.launch --nproc_per_node=1
```

To evaluate the LPIPS score

```bash
python -m torch.distributed.launch --nproc_per_node=1 
```

## Training

### Data preparing

When training FFHQ , The data structure is like this:

```
ffhq
├── 00001.png
├── 00002.png
│   ...
```

### FFHQ

To train a new model of **EmbedStyleSwin-FFHQ-512** from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_styleswin.py --batch 2 --path /path_to_ffhq_1024 --checkpoint_path /tmp --sample_path /tmp --size 1024 --D_lr 0.0002 --D_sn --ttur --eval_gt_path /path_to_ffhq_real_images_50k --lr_decay --lr_decay_start_steps 600000
```
or
```bash
bash run.sh
```
**Notice**: When training on A100(40GB) GPUs, you could add `--use_checkpoint` to save GPU memory. 
Besides, we evaluate the LPIPS score every 10000 steps during training.

## Qualitative Results

Image samples Restoration of --xx-dataset--  by EmbedStyleSwin:

(traing progress not finish yet)
![](imgs/160600.png)


## Citing EmbedStyleSwin

```
@misc{EmbedStyleSwin,
      title={EmbedStyleSwin: Embed Transformer-based GAN for Blind Face Restoration}, 
      author={-----},
      year={2022},
      eprint={0.0},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Responsible AI Considerations

Our work 

## Acknowledgements

This code borrows heavily from [GPEN](github.com) 、 [StyleSwin](github.com) 、[stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We also thank the contributors of code [Positional Encoding in GANs](https://github.com/open-mmlab/mmgeneration/blob/master/configs/positional_encoding_in_gans/README.md), [DiffAug](https://github.com/mit-han-lab/data-efficient-gans), [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) and [GIQA](https://github.com/cientgu/GIQA).

## Maintenance

This is the codebase for our research work. Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact [a@bnu.edu.cn](mail@mail.com) or [mail](mail@mail.com).


## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.


