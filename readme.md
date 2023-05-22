# Detection, classification and recognition of brazilian vertical traffic sings and lights from a car using single shot multibox detector.
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

## Configuration
We use YAML for configuration management. See `configs/*/*.yaml` for examples.
You can modify the settings as needed.

## Training
```bash
python train.py
```

## For example
```bash
python train.py -mi 0 (mobilenetv2_ssdlite)
python train.py -mi 1 (mobilenetv3large_ssdlite)
python train.py -mi 2 (mobilenetv3small_ssdlite)
```

To visualize training progress using TensorBoard:
```bash
tensorboard --logdir <LOG_DIRECTORY>
```