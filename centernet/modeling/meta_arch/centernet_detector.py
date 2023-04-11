import math
import json
import numpy as np
import torch
from torch import nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone, build_proposal_generator
from detectron2.modeling import detector_postprocess
from detectron2.structures import ImageList

@META_ARCH_REGISTRY.register()
class CenterNetDetector(nn.Module): #这是一个类，继承自nn.Module，nn.Module是所有神经网络模块的基类，里面可以放各种各样的网络层
    def __init__(self, cfg):    #初始化函数，cfg是配置文件
        super().__init__() #调用父类的构造函数
        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD #获取像素均值和标准差
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)) #将像素均值转换为张量
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)) #将像素标准差转换为张量
        
        self.backbone = build_backbone(cfg) #构建backbone
        self.proposal_generator = build_proposal_generator( #根据backbone的output_shape构建proposal_generator
            cfg, self.backbone.output_shape()) # TODO: change to a more precise name
    
    #前向传播函数，输入为batched_inputs，是一个列表，每个元素是一个字典，字典中包含了图像的信息，如图像的路径、图像的尺寸等，这些信息都是在dataset中定义的
    def forward(self, batched_inputs):  #训练的时候要调用这个函数来计算损失函数，测试的时候，需要调用这个函数来计算预测结果
        if not self.training:   #如果不是训练阶段，就调用inference函数来计算预测结果
            return self.inference(batched_inputs)   #返回预测结果
        images = self.preprocess_image(batched_inputs)  #如果是训练阶段，就调用preprocess_image函数来对图像进行预处理
        features = self.backbone(images.tensor) #将预处理后的图像输入backbone，得到backbone的输出
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] #将ground truth的信息转换为张量

        _, proposal_losses = self.proposal_generator(   #调用proposal_generator来计算损失函数
            images, features, gt_instances) #输入图像、backbone的输出、ground truth的信息，得到损失函数
        return proposal_losses


    @property   
    def device(self):
        return self.pixel_mean.device


    @torch.no_grad()    #不计算梯度
    def inference(self, batched_inputs, do_postprocess=True):   #计算预测结果
        images = self.preprocess_image(batched_inputs)  #预处理图像
        inp = images.tensor #获取预处理后的图像
        features = self.backbone(inp)   #将预处理后的图像输入backbone，得到backbone的输出
        proposals, _ = self.proposal_generator(images, features, None)  #调用proposal_generator来计算预测结果

        processed_results = []  #存储预测结果
        for results_per_image, input_per_image, image_size in zip( #遍历图像库，zip函数将可迭代的对象作为参数，将其对应元素打包成一个个元组，返回这些元组组成的列表
            proposals, batched_inputs, images.image_sizes):
            if do_postprocess:  #如果需要后处理
                height = input_per_image.get("height", image_size[0])   #获取图像的高度
                width = input_per_image.get("width", image_size[1]) #获取图像的宽度
                r = detector_postprocess(results_per_image, height, width)  #调用后处理函数
                processed_results.append({"instances": r})  #将后处理后的结果存储到processed_results中
            else:
                r = results_per_image   #如果不需要后处理，就直接将预测结果存储到processed_results中
                processed_results.append(r) #将预测结果存储到processed_results中
        return processed_results    #返回预测结果

    def preprocess_image(self, batched_inputs):  #预处理图像
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]   #将图像转换为张量
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]   #对图像进行归一化
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)    #将图像转换为ImageList
        return images   #返回预处理后的图像
