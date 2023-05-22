# PyTorch SSD
PyTorch implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

## Results
### PASCAL VOC
* Training: 07+12 trainval
* Evaluation: 07 test

| Model                | Input size | mAP<sub>0.5</sub> | Configuration                                                                |
|----------------------|:----------:|:-----------------:|------------------------------------------------------------------------------|
| SSD300               | 300        | 77.1              | [configs/voc/ssd300.yaml](configs/voc/ssd300.yaml)                           |
| SSD512               | 512        | 79.4              | [configs/voc/ssd512.yaml](configs/voc/ssd512.yaml)                           |
| MobileNetV2 SSDLite  | 320        | 70.7              | [configs/voc/mobilenetV2_ssdlite.yaml](configs/voc/mobilenetV2_ssdlite.yaml) |

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
python train.py --cfg <CONFIG_FILE> --logdir <LOG_DIRECTORY>

# For example, to train SSD300 on PASCAL VOC:
python train.py --cfg configs/voc/ssd300.yaml --logdir runs/voc_ssd300/exp0/
```
To visualize training progress using TensorBoard:
```bash
tensorboard --logdir <LOG_DIRECTORY>
```
An interrupted training can be resumed by:
```bash
# Run train.py with --resume to restore the latest saved checkpoint file in the log directory.
python train.py --cfg <CONFIG_FILE> --logdir <LOG_DIRECTORY> --resume
```

## Evaluation
### PASCAL VOC
```bash
python eval.py --cfg <CONFIG_FILE> --pth <LOG_DIRECTORY>/best.pth --dataset datasets/voc/val.json
```