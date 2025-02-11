from easydict import EasyDict

cfg = EasyDict()

cfg.device='cpu'
cfg.strides = [8, 16, 32]
cfg.anchors = [[[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
               [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
               [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]]
