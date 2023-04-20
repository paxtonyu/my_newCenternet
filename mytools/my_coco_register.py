from detectron2.data.datasets import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

coco_50 = ['person','backpack','umbrella','handbag','tie','suitcase',                     
 'bicycle', 'cat','dog', 'bottle','wine glass','cup','fork','knife','spoon','bowl',                                    
 'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
 'cake', 'chair','couch','potted plant','bed','dining table','toilet', 'tv','laptop',
 'mouse','remote','keyboard','cell phone', 'microwave','oven','toaster','sink',
 'refrigerator', 'book','clock','vase','scissors','teddy bear','hair drier','toothbrush',]

def register_my_coco():
    register_coco_instances_v2("my_indoor_dataset_train", {}, "datasets/coco_my/annotations/my_instances_train2017.json", "datasets/coco_my/train2017")
    register_coco_instances_v2("my_indoor_dataset_val", {}, "datasets/coco_my/annotations/my_instances_val2017.json", "datasets/coco_my/val2017")

def register_coco_instances_v2(name, metadata, json_file, image_root):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )
    MetadataCatalog.get(name).thing_classes = coco_50