# Test code for PCNN
## Prepare
* install the latest PyTorch (1.1.0)
* download this code (pretrained model included)
* get the ImageNet validation set ready (you may need the script *valprep.sh* to pre-process the val set)

## Check whether the weights and activation is binary
* run ```python weight_summary.py``` to check whether the weights are binary
* check the code in the module ```BinConv2d``` to see whether the input for convolution is binary

## Evaluation 
Modifications in the script ```evaluate_imagenet.sh```:
* modify the **PATH to your ImageNet dataset** 
* modify the **batchsize** (default: 256) according to your hardware (at least one GPU is requried)

Run the script ```evaluate_imagenet.sh``` and the accuracy on validation set is around **57.30**. 

## Please cite

```
@inproceedings{gu2019projection,
  title={Projection Convolutional Neural Networks for 1-bit CNNs via Discrete Back Propagation},
  author={Gu, Jiaxin and Li, Ce and Zhang, Baochang and Han, Jungong and Cao, Xianbin and Liu, Jianzhuang and Doermann, David},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={8344--8351},
  year={2019}
}
```

