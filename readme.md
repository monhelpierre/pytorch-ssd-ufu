# Recognition of Brazilian vertical traffic sings and lights from a car using Single Shot Multi box Detector (SSD).
PyTorch implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) from orignal repository (https://github.com/biyoml/PyTorch-SSD).

## Results

* Training: 07 train
* Evaluation: 02 val
* Testing: 01 test

Related dataset : (https://data.mendeley.com/datasets/jbpsr4fvg9/1)

Related paper : (https://journals-sol.sbc.org.br/index.php/jbcs/article/view/3678)

| Model                      | Input size | mAP<sub>0.5</sub> | Configuration                                                        |
|----------------------------|:----------:|:-----------------:|----------------------------------------------------------------------|
| MobileNetV2 SSDLite        | 320        | 87.4%             | [configs/mobilenetV2_ssdlite.yaml]                                   |
| MobileNetV3 Large SSDLite  | 320        | 84.5%             | [configs/mobilenetV3Small_ssdlite.yaml]                              |
| MobileNetV3 Small SSDLite  | 320        | 46.0%             | [configs/mobilenetV3Large_ssdlite.yaml]                              |
| MobileNetV2 SSDLite        | 512        | 77.6%             | [configs/mobilenetV2_ssdlite.yaml]                                   |
| MobileNetV3 Small SSDLite  | 512        | 54.6%             | [configs/mobilenetV3Small_ssdlite.yaml]                              |
| MobileNetV3 Small SSDLite  | 128        | 11.2%             | [configs/mobilenetV3Small_ssdlite.yaml]                              |


## Requirements
* Python ≥ 3.6
* Install libraries: `pip install -r requirements.txt`

## Configuration
We use YAML for configuration management. See `configs/*.yaml` for examples.
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

## Class description
| Label      | English Description             | Portuguese Description                      |
|------------|:--------------------------------|:--------------------------------------------|
| 000        | Stop sign                       | Parada obrigatória                          |
| 001        | Give way                        | Dê a preferência                            |
| 003        | No left turn                    | Proíbido virar à esquerda                   |
| 004        | No right turn                   | Proíbido virar à direita                    |
| 007        | No park                         | Proíbido estacionar                         |
| 008        | Regular park                    | Estacionamento regularizdo                  |
| 009        | No park and stop                | Proíbido parar e estacionar                 |
| 023        | Speed limit                     | Limite de velocidade permitida              |
| 025        | Road hump                       | Lombada                                   |
| 028        | Direction of the way circulation| Sentido de circulação da via                |
| 035        | Truck keep right                | Veículo pesados, mantenham-se à direita     |
| 040        | Bus route                       | Fôaixa de ônibus                            |
| 042        | Cycling                         | Bicycleta permitida                         |
| 051        | Yellow light                    | Atenção veículos                            |
| 052        | Red light                       | Parada de veículos                          |
| 053        | Green light                     | Veículos podem seguir                       |


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


To cite this work:
```
@misc{monhel2023,
  author = {Monhel, Maudoony Pierre},
  title = {Recognition of Brazilian Traffic signs and lights using Single Shot Multi box Detector},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/monhelpierre/pytorch-ssd-ufu/}}
}
```
