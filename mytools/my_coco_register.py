from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

coco_50 = ['person','backpack','umbrella','handbag','tie','suitcase',                     
 'bicycle', 'cat','dog', 'bottle','wine glass','cup','fork','knife','spoon','bowl',                                    
 'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
 'cake', 'chair','couch','potted plant','bed','dining table','toilet', 'tv','laptop',
 'mouse','remote','keyboard','cell phone', 'microwave','oven','toaster','sink',
 'refrigerator', 'book','clock','vase','scissors','teddy bear','hair drier','toothbrush',]

def register_my_coco():
    register_coco_instances("my_indoor_dataset_train", {}, "datasets/coco_my/annotations/my_instances_train2017.json", "datasets/coco_my/train2017")
    register_coco_instances("my_indoor_dataset_val", {}, "datasets/coco_my/annotations/my_instances_val2017.json", "datasets/coco_my/val2017")
    register_coco_instances("rebar_train", {}, "datasets/rebar/annotations/train.json", "datasets/rebar/images/train_data")
    register_coco_instances("rebar_val", {}, "datasets/rebar/annotations/val.json", "datasets/rebar/images/val_data")
    register_coco_instances("nurse_train", {}, "datasets/nurse6/annotations/train.json", "datasets/nurse6/train")
    register_coco_instances("nurse_val", {}, "datasets/nurse6/annotations/val.json", "datasets/nurse6/val")