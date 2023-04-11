import math
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, get_norm
from detectron2.config import configurable
from ..layers.deform_conv import DFConv2d

__all__ = ["CenterNetHead"] 

class Scale(nn.Module): #Scale是一个可学习的参数，用于调整bbox的大小
    def __init__(self, init_value=1.0): #init_value是初始值
        super(Scale, self).__init__()   #super()函数是用于调用父类(超类)的一个方法，这里是调用nn.Module的__init__()方法
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))  #nn.Parameter()是一个tensor，但是当作模型的一部分进行训练，即可学习的参数

    def forward(self, input):   #forward()方法是必须要有的，这里是将输入的input乘以scale
        return input * self.scale

class CenterNetHead(nn.Module): #CenterNetHead是CenterNet的头部，用于预测bbox和分类
    @configurable   #configurable是一个装饰器，用于将类的参数传入类中
    def __init__(self,  
        # input_shape: List[ShapeSpec],
        in_channels,    #in_channels是输入的通道数
        num_levels,    #num_levels是输入的特征图的数量
        *,  #*表示后面的参数必须使用关键字传参
        num_classes=80, #num_classes是类别数
        with_agn_hm=False,  #with_agn_hm表示是否使用agn_hm，agn_hm是用于预测是否为人的heatmap
        only_proposal=False,    #only_proposal表示是否只预测proposal
        norm='GN',  #norm是归一化方法，这里使用的是GroupNorm，GroupNorm是对通道进行归一化
        num_cls_convs=4,    #num_cls_convs是分类的卷积层数
        num_box_convs=4,    #num_box_convs是bbox的卷积层数
        num_share_convs=0,  #num_share_convs是共享的卷积层数
        use_deformable=False,   #use_deformable表示是否使用可变形卷积
        prior_prob=0.01):   #prior_prob是先验概率
        super().__init__()  #调用父类的__init__()方法
        self.num_classes = num_classes  #将num_classes赋值给self.num_classes
        self.with_agn_hm = with_agn_hm  #将with_agn_hm赋值给self.with_agn_hm
        self.only_proposal = only_proposal  #将only_proposal赋值给self.only_proposal
        self.out_kernel = 3 #out_kernel是输出的卷积核大小

        head_configs = {    #head_configs是头部的配置
            "cls": (num_cls_convs if not self.only_proposal else 0, \
                use_deformable),    #cls是分类
            "bbox": (num_box_convs, use_deformable),    #bbox是bbox
            "share": (num_share_convs, use_deformable)} #share是共享的

        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, \
        #     "Each level must have the same channel!"
        # in_channels = in_channels[0]
        channels = {
            'cls': in_channels, 
            'bbox': in_channels,
            'share': in_channels,
        }
        for head in head_configs:   # 遍历head_configs
            tower = []  # tower是一个列表，列表中的元素是nn.Sequential()的参数
            # nn.Sequential()是一个顺序容器，网络中的层将按照传递给构造函数的顺序被添加到计算图中
            # 当传递给forward()的输入数据流经这些层时，将产生输出数据
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):  
                if use_deformable and i == num_convs - 1:   #如果使用可变形卷积且是最后一层卷积
                    conv_func = DFConv2d    #使用DFConv2d，DFConv2d是可变形卷积
                else:
                    conv_func = nn.Conv2d   #否则使用nn.Conv2d，nn.Conv2d是普通卷积
                tower.append(conv_func( #添加卷积层，conv_func是卷积层，是nn.Conv2d()或DFConv2d()
                        in_channels if i == 0 else channel, 
                        channel,    
                        kernel_size=3, stride=1,    #卷积核大小为3，步长为1
                        padding=1, bias=True    #padding为1，bias为True
                ))  # nn.Conv2d()是一个二维卷积层
                # in_channels是输入的通道数，channel是输出的通道数，kernel_size是卷积核大小
                # stride是步长，padding是填充，bias是是否使用偏置
                if norm == 'GN' and channel % 32 != 0:  #如果使用GroupNorm且channel不能被32整除
                    tower.append(nn.GroupNorm(25, channel)) #使用nn.GroupNorm，GroupNorm是对通道进行归一化
                elif norm != '':    #否则如果norm不为空
                    tower.append(get_norm(norm, channel))   #使用get_norm()方法，get_norm()方法是获取归一化方法
                tower.append(nn.ReLU()) #添加激活函数
            self.add_module('{}_tower'.format(head),    #添加模块，模块名为head_tower
                            nn.Sequential(*tower))  #添加tower，tower是一个列表，列表中的元素是nn.Sequential()的参数

        self.bbox_pred = nn.Conv2d( #添加bbox_pred，bbox_pred是用于预测bbox的卷积层
            in_channels, 4, kernel_size=self.out_kernel,  #输入通道数为in_channels，输出通道数4，卷积核大小为out_kernel
            stride=1, padding=self.out_kernel // 2  #步长为1，padding为out_kernel // 2
        )

        self.scales = nn.ModuleList(    #添加scales，scales是一个列表，列表中的元素是Scale()的参数
            [Scale(init_value=1.0) for _ in range(num_levels)]) #Scale()是用于缩放特征图的类

        for modules in [    #遍历[cls_tower, bbox_tower, share_tower, bbox_pred]
            self.cls_tower, self.bbox_tower,    #cls_tower是分类的卷积层，bbox_tower是bbox的卷积层
            self.share_tower,   #share_tower是共享的卷积层
            self.bbox_pred, #bbox_pred是用于预测bbox的卷积层
        ]:
            for l in modules.modules(): #遍历modules.modules()
                if isinstance(l, nn.Conv2d):    #如果l是nn.Conv2d()，nn.Conv2d()是一个二维卷积层
                    torch.nn.init.normal_(l.weight, std=0.01)   #使用torch.nn.init.normal_()初始化权重，std是标准差
                    torch.nn.init.constant_(l.bias, 0)  #使用torch.nn.init.constant_()初始化偏置，bias是偏置
        
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)    # bbox_pred.bias是偏置，初始化为8
        prior_prob = prior_prob # prior_prob是先验概率
        bias_value = -math.log((1 - prior_prob) / prior_prob)   # bias_value计算公式为-log((1 - prior_prob) / prior_prob)

        if self.with_agn_hm:
            self.agn_hm = nn.Conv2d(
                in_channels, 1, kernel_size=self.out_kernel,
                stride=1, padding=self.out_kernel // 2
            )
            torch.nn.init.constant_(self.agn_hm.bias, bias_value)
            torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(
                in_channels, self.num_classes,
                kernel_size=cls_kernel_size, 
                stride=1,
                padding=cls_kernel_size // 2,
            )

            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            # 'input_shape': input_shape,
            'in_channels': [s.channels for s in input_shape][0],
            'num_levels': len(input_shape),
            'num_classes': cfg.MODEL.CENTERNET.NUM_CLASSES,
            'with_agn_hm': cfg.MODEL.CENTERNET.WITH_AGN_HM,
            'only_proposal': cfg.MODEL.CENTERNET.ONLY_PROPOSAL,
            'norm': cfg.MODEL.CENTERNET.NORM,
            'num_cls_convs': cfg.MODEL.CENTERNET.NUM_CLS_CONVS,
            'num_box_convs': cfg.MODEL.CENTERNET.NUM_BOX_CONVS,
            'num_share_convs': cfg.MODEL.CENTERNET.NUM_SHARE_CONVS,
            'use_deformable': cfg.MODEL.CENTERNET.USE_DEFORMABLE,
            'prior_prob': cfg.MODEL.CENTERNET.PRIOR_PROB,
        }
        return ret

    def forward(self, x):
        clss = []
        bbox_reg = []
        agn_hms = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if not self.only_proposal:
                clss.append(self.cls_logits(cls_tower))
            else:
                clss.append(None)

            if self.with_agn_hm:
                agn_hms.append(self.agn_hm(bbox_tower))
            else:
                agn_hms.append(None)
            reg = self.bbox_pred(bbox_tower)
            reg = self.scales[l](reg)
            bbox_reg.append(F.relu(reg))
        
        return clss, bbox_reg, agn_hms