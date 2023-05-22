# PyTorch SSD
PyTorch implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

## Results
### PASCAL VOC
* Training: 07+12 trainval
* Evaluation: 07 test

| Model                      | Input size | mAP<sub>0.5</sub> | Configuration                                                                |
|----------------------------|:----------:|:-----------------:|------------------------------------------------------------------------------|
| MobileNetV2 SSDLite        | 320        | 80.5              | [configs/mobilenetV2_ssdlite.yaml](configs/mobilenetV2_ssdlite.yaml) |
| MobileNetV3 Small SSDLite  | 320        | 82.7              | [configs/mobilenetV2_ssdlite.yaml](configs/mobilenetV2_ssdlite.yaml) |
| MobileNetV3 Large SSDLite  | 320        | 50.9              | [configs/mobilenetV2_ssdlite.yaml](configs/mobilenetV2_ssdlite.yaml) |

## Requirements
* Python ≥ 3.6
* Install libraries: `pip install -r requirements.txt`

## Data Preparation
### PASCAL VOC
```bash
cd datasets/voc/
python prepare.py --root VOCdevkit/
```
## Configuration
We use YAML for configuration management. See `configs/*/*.yaml` for examples.
You can modify the settings as needed.

## Training
```bash
python train.py
```

To visualize training progress using TensorBoard:
```bash
tensorboard --logdir <LOG_DIRECTORY>
```