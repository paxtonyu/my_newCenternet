from detectron2.data.datasets import register_coco_instances

def register_my_coco():
    register_coco_instances("my_indoor_dataset_train", {}, "datasets/coco_my/annotations/instances_train2017.json", "datasets/coco_my/train2017")
    register_coco_instances("my_indoor_dataset_val", {}, "datasets/coco_my/annotations/instances_val2017.json", "datasets/coco_my/val2017")