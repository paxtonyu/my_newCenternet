# Indoor object detection based on multi-scale feature fusion and attention mechanism

基于centernet补充添加Attention和DLA

## Eval on my indoor dataset val
>CenterNet-S4_DLA_8x

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 40.252 | 56.992 | 44.082 | 22.046 | 42.783 | 53.771 |

>CenterNet2_R50_1x

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 39.988 | 56.057 | 44.053 | 19.215 | 42.457 | 53.871 |

>CenterNet2_DLA-BiFPN-P3_24x

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 43.046 | 59.372 | 46.767 | 20.158 | 46.179 | 58.759 |

## 使用nohup
`nohup python ./train_net.py >> ./mylog/train.log 2>&1 &`

## 使用tensorboard
`tensorboard --logdir=./output/MY/folderpath --port=6006`

## Probabilistic two-stage detection
Two-stage object detectors that use class-agnostic one-stage detectors as the proposal network.


<p align="center"> <img src='docs/centernet2_teaser.jpg' align="center" height="150px"> </p>

> [**Probabilistic two-stage detection**](http://arxiv.org/abs/2103.07461),            
> Xingyi Zhou, Vladlen Koltun, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 2103.07461](http://arxiv.org/abs/2103.07461))*         

Contact: [zhouxy@cs.utexas.edu](mailto:zhouxy@cs.utexas.edu). Any questions or discussions are welcomed! 

## Summary

- Two-stage CenterNet: First stage estimates object probabilities, second stage conditionally classifies objects.

- Resulting detector is faster and more accurate than both traditional two-stage detectors (fewer proposals required), and one-stage detectors (lighter first stage head).

- Our best model achieves 56.4 mAP on COCO test-dev.

- This repo also includes a detectron2-based CenterNet implementation with better accuracy (42.5 mAP at 70FPS) and a new FPN version of CenterNet (40.2 mAP with Res50_1x).

## License

Our code is under [Apache 2.0 license](LICENSE). `centernet/modeling/backbone/bifpn_fcos.py` are from [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), which follows the original [non-commercial license](https://github.com/aim-uofa/AdelaiDet/blob/master/LICENSE).

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2021probablistic,
      title={Probabilistic two-stage detection},
      author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:2103.07461},
      year={2021}
    }
