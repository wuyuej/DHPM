# DHPM
**Paper:** Dense Hybrid Proposal Modulation for Lane Detection

*This repository is based on BezierLaneNet, we have improved it by applying availability constraint, diversity constraint and discrimination constraint, so users only need to download the code of BezierLaneNet, and then download "hungarian_bezier_loss.py" in this repository to replace it. For different datasets, please modify the values in lines 12 and 13.

*To experiment on CurveLanes, you need to first use convert_curvelanes.py and convert_curvelanes.py under https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2 to generate segmentation and labels. Thanks a lot for their help.

*BezierLaneNet's address is https://github.com/voldemortX/pytorch-auto-drive

## Performance
### TuSimple performance (best):
| method | backbone | accuracy | FP | FN |
| :---: | :---: | :---: | :---: | :---: |
|BezierLaneNet| resnet18 | 95.41 | 5.30 | 4.60 |
|BezierLaneNet| resnet34 | 95.65 | 5.10 | 3.90 |
| Ours | resnet18 | 95.61 | 5.30 | 3.50 |
| Ours | resnet34 | 95.87 | 5.00 | 3.40 |

### CULane performance (best):
| method | backbone | F1 |
| :---: | :---: | :---: |
|BezierLaneNet| resnet18 | 73.67 |
|BezierLaneNet| resnet34 | 75.57 |
| Ours | resnet18 | 74.59 |
| Ours | resnet34 | 76.21 |

### LLAMAS performance (best):
| method | backbone | F1 |
| :---: | :---: | :---: |
|BezierLaneNet| resnet18 | 94.91 |
|BezierLaneNet| resnet34 | 95.17 |
| Ours | resnet18 | 95.15 |
| Ours | resnet34 | 95.30 |

### CurveLanes performance (best):
| method | backbone | F1 |
| :---: | :---: | :---: |
|BezierLaneNet| resnet | 74.56 |
| Ours | resnet | 75.03 |


## Citation

```
@inproceedings{BezierLaneNet,
  title={Rethinking efficient lane detection via curve modeling},
  author={Feng, Zhengyang and Guo, Shaohua and Tan, Xin and Xu, Ke and Wang, Min and Ma, Lizhuang},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}
@inproceedings{culane,
  author    = {Xingang Pan and
               Jianping Shi and
               Ping Luo and
               Xiaogang Wang and
               Xiaoou Tang},
  title     = {Spatial as Deep: Spatial {CNN} for Traffic Scene Understanding},
  booktitle = {{AAAI}},
  pages     = {7276--7283},
  year      = {2018}
}
@misc{tusimple,
   author = {TuSimple},
   title = {Tusimple benchmark.},
   howpublished = {\url{https://github.com/TuSimple/tusimple-benchmark/}},
   year={Accessed September, 2020.}
}
@inproceedings{llamas,
  title={Unsupervised Labeled Lane Markers Using Maps},
  author={Behrendt Karsten and Soussan Ryan},
  booktitle={ICCVW},
  pages={832--839},
  year={2019}
}
@inproceedings{CurveLanes,
  author    = {Hang Xu and
               Shaoju Wang and
               Xinyue Cai and
               Wei Zhang and
               Xiaodan Liang and
               Zhenguo Li},
  title     = {CurveLane-NAS: Unifying Lane-Sensitive Architecture Search and Adaptive Point Blending},
  booktitle = {ECCV},
  pages     = {689--704},
  year      = {2020}
}
```
