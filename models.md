SSDLite(
  (0): MobileNetV2(
    (trunk): MobileNetV2(
      (features): Module(
        (0): Conv2dNormActivation(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (3): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (4): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (6): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (8): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (9): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (10): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (11): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (12): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (13): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (14): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (15): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (16): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (17): Module(
          (conv): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (18): Conv2dNormActivation(
          (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
    )
    (extra_layers): ModuleList(
      (0): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (1): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (2): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (3): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
    )
  )
  (1): _Heads(
    (classifincation_heads): ModuleList(
      (0): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(576, 68, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280, bias=False)
          (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(1280, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(512, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(256, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (4): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(256, 68, kernel_size=(1, 1), stride=(1, 1))
      )
      (5): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(128, 68, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (regression_heads): ModuleList(
      (0): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(576, 16, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280, bias=False)
          (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(1280, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(256, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (4): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))
      )
      (5): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
Number of parameters : 3.286.326
Number of trainable parameters : 3.286.326



SSDLite(
  (0): MobileNetV3Large(
    (trunk): MobileNetV3(
      (features): Module(
        (0): Conv2dNormActivation(
          (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): Hardswish()
        )
        (1): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
              (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (2): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
              (1): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (3): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
              (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (4): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
              (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (5): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
              (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (6): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
              (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (7): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
              (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (8): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(80, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=200, bias=False)
              (1): BatchNorm2d(200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(200, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (9): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
              (1): BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (10): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
              (1): BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (11): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
              (1): BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(480, 120, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(120, 480, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (12): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
              (1): BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (13): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
              (1): BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (14): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (15): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (16): Conv2dNormActivation(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): Hardswish()
        )
      )
    )
    (extra_layers): ModuleList(
      (0): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(240, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (1): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(480, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=120, bias=False)
          (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(120, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (2): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(240, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=120, bias=False)
          (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(120, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (3): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(240, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(60, 60, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=60, bias=False)
          (1): BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(60, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
    )
  )
  (1): _Heads(
    (classifincation_heads): ModuleList(
      (0): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(960, 68, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(960, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
          (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(480, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(240, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (4): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(240, 68, kernel_size=(1, 1), stride=(1, 1))
      )
      (5): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)
          (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(120, 68, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (regression_heads): ModuleList(
      (0): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(960, 16, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(960, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
          (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(480, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(240, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (4): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
          (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(240, 16, kernel_size=(1, 1), stride=(1, 1))
      )
      (5): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)
          (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(120, 16, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
Number of parameters : 3.881.522
Number of trainable parameters : 3.881.522



SSDLite(
  (0): MobileNetV3Small(
    (trunk): MobileNetV3(
      (features): Module(
        (0): Conv2dNormActivation(
          (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): Hardswish()
        )
        (1): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
              (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (2): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(16, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
              (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (3): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(88, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
              (1): BatchNorm2d(88, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (4): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(96, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=96, bias=False)
              (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(96, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (5): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
              (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(64, 240, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (6): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
              (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(64, 240, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (7): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
              (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(120, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (8): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(48, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=144, bias=False)
              (1): BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(40, 144, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (9): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
              (1): BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(288, 72, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(72, 288, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (10): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (11): Module(
          (block): Module(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
              (2): Hardswish()
            )
            (2): SqueezeExcitation(
              (avgpool): AdaptiveAvgPool2d(output_size=1)
              (fc1): Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
              (fc2): Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
              (activation): ReLU()
              (scale_activation): Hardsigmoid()
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            )
          )
        )
        (12): Conv2dNormActivation(
          (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): Hardswish()
        )
      )
    )
    (extra_layers): ModuleList(
      (0): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(144, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (1): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(288, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
          (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(72, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (2): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(144, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
          (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(72, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (3): _ExtraBlock(
        (0): ConvBNReLU(
          (0): Conv2d(144, 36, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=36, bias=False)
          (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): ConvBNReLU(
          (0): Conv2d(36, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
    )
  )
  (1): _Heads(
    (classifincation_heads): ModuleList(
      (0): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(144, 68, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(576, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
          (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(288, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(144, 102, kernel_size=(1, 1), stride=(1, 1))
      )
      (4): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(144, 68, kernel_size=(1, 1), stride=(1, 1))
      )
      (5): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
          (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(72, 68, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (regression_heads): ModuleList(
      (0): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(144, 16, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
          (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(288, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1))
      )
      (4): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(144, 16, kernel_size=(1, 1), stride=(1, 1))
      )
      (5): _SSDLiteHead(
        (0): ConvBNReLU(
          (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
          (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(72, 16, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
Number of parameters : 1.304.522
Number of trainable parameters : 1.304.522