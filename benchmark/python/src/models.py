#! /usr/bin/env python3
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import torch.onnx
import torchvision
from mgeconvert.converters.onnx_to_mge import *

megcc_path = Path(
    os.path.split(os.path.realpath(__file__))[0]
).parent.parent.parent.absolute()
default_gen_path = "{}/benchmark/model/generated_models".format(megcc_path)


class Model:
    name = None
    torch_model = None
    input_shape = []

    def __init__(self, name, torch_model, input_shape):
        self.name = name
        self.torch_model = torch_model
        self.input_shape = input_shape


class AllModel:
    models = []
    # model src from onnx
    def __init__(self):
        # pytorch model
        self.models.append(
            Model(
                "mobilenetv2",
                torchvision.models.mobilenetv2.mobilenet_v2(),
                [1, 3, 224, 224],
            )
        )
        self.models.append(
            Model(
                "efficientnetb0",
                torchvision.models.efficientnet.efficientnet_b0(),
                [1, 3, 256, 256],
            )
        )
        self.models.append(
            Model(
                "shufflenetv2",
                torchvision.models.shufflenetv2.shufflenet_v2_x0_5(),
                [1, 3, 224, 224],
            )
        )
        self.models.append(
            Model("resnet18", torchvision.models.resnet.resnet18(), [1, 3, 224, 224])
        )
        self.models.append(
            Model("resnet50", torchvision.models.resnet.resnet50(), [1, 3, 224, 224])
        )
        self.models.append(
            Model("vgg11", torchvision.models.vgg.vgg11(), [1, 3, 224, 224])
        )
        self.models.append(
            Model("vgg16", torchvision.models.vgg.vgg16(), [1, 3, 224, 224])
        )

    def get_all_onnx_models(self, output_dir=default_gen_path):
        if not os.path.exists(output_dir) or os.path.isfile(output_dir):
            os.makedirs(output_dir)
        for model in self.models:
            output = "{}/{}.onnx".format(output_dir, model.name)
            logging.debug("get model file from torchvision to: {}".format(output))
            net = model.torch_model
            net.eval()
            input_data = torch.randn(model.input_shape)
            torch.onnx.export(
                net,
                input_data,
                output,
                export_params=True,
                opset_version=12,
                input_names=["data"],
                output_names=["ret"],
            )

    def convert_to_mge(self, output_dir=default_gen_path):
        for model in self.models:
            input = "{}/{}.onnx".format(output_dir, model.name)
            output = "{}/{}.mge".format(output_dir, model.name)
            onnx_to_mge(input, output)

    def make(self, model_dir=""):
        if model_dir != "":
            self.get_all_onnx_models(model_dir)
            self.convert_to_mge(model_dir)
        else:
            self.get_all_onnx_models()
            self.convert_to_mge()
