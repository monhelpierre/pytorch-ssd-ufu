# Detection, classification and recognition of brazilian vertical traffic sings and lights from a car using single shot multibox detector.
PyTorch implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) from orignal repository (https://github.com/biyoml/PyTorch-SSD).

## Results

* Training: 07 train
* Evaluation: 02 val
* Testing: 01 test

| Model                      | Input size | mAP<sub>0.5</sub> | Configuration                                                                |
|----------------------------|:----------:|:-----------------:|------------------------------------------------------------------------------|
| MobileNetV2 SSDLite        | 320        | 87.4              | [configs/mobilenetV2_ssdlite.yaml](configs/mobilenetV2_ssdlite.yaml) |
| MobileNetV3 Small SSDLite  | 320        | 84.5              | [configs/mobilenetV2_ssdlite.yaml](configs/mobilenetV2_ssdlite.yaml) |
| MobileNetV3 Large SSDLite  | 320        | 46.0              | [configs/mobilenetV2_ssdlite.yaml](configs/mobilenetV2_ssdlite.yaml) |

## Requirements
* Python ≥ 3.6
* Install libraries: `pip install -r requirements.txt`

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

To test from a video:
```bash
python test.py -mi 0 --video "/path/to/video/" --save "/path/to/save/video/output/" (mobilenetv2_ssdlite)
```

To test from an image:
```bash
python test.py -mi 0 --image "/path/to/image/" --save "/path/to/save/image/output/" (mobilenetv2_ssdlite)
```

To test from the test dataset where output images will be saved in logs folder:
```bash
python test.py -mi 0 (mobilenetv2_ssdlite)
```

CLASS  -   DESCRIPTION
-----------------------------------
000        -   Stop sign
001        -   Give away
003        -   No left turn
004        -   No right turn
007        -   No park
008        -   Regular park
009        -   No park and stop
023        -   Speed limit
025        -   Road hump
028        -   Sense of the way circulation
035        -   Truck keep right
040        -   Bus route
042        -   Cycling
051        -   Yellow light
052        -   Red light
053        -   Green light