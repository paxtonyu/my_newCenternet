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