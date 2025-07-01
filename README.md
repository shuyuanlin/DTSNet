# DTSNet
DTSNet: A denoising teacher-student network with reverse distillation for anomaly detection

## DTSNet implementation
Code implementation of ICME 2025 paper: DTSNet: A denoising teacher-student network with reverse distillation for anomaly detection.

## Requirements

```
pip install -r requirements.txt
```

## Datasets

Download Mvtec dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

Download VisA dataset from [URL](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar).

Download BTAD dataset from [URL](https://avires.dimi.uniud.it/papers/btad/btad.zip).

## Test pretrained model
The pretrained weights can be found [here](TBD).

```shell
python inference.py
```
## Train
To train and test the DTSNet method on MVTEC/VisA/BTAD, please run:

```shell
python main_mvtec.py
```

```shell
python main_visa.py
```

```shell
python main_btad.py
```

## ...


## Citation
```
@article{lin2025dtsnet,
  title = {DTSNet: A denoising teacher-student network with reverse distillation for anomaly detection},
  author = {Taixiang Lin, Shuyuan Lin*, Yanjie Liang, Rong Rong, Yang Lu},
  journal = {Proceedings of the IEEE International Conference on Multimedia and Expo},
  pages = {1--6},
  year = {2025},
}
```