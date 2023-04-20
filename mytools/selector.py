from pycocotools.coco import COCO
import os
import numpy as np
import json
import shutil

# 需要设置的路径
dataDir = './coco/'             #原始数据集路径
savepath = "./coco_my/"         #保存数据集路径
img_dir = savepath + 'images/'
anno_dir = savepath + 'annotations/'
datasets_list = ['val2017','train2017']     #待处理数据集列表

'''
目录格式如下：(dataDir下)
$COCO_PATH
----|annotations
----|train2017
----|val2017
'''
#由于一张图片中有很多种类目标，直接选则需要的种类可能不符合场景需求
#可以通过选则不可能在目标场景中出现的类别进行排除，取剩余数据集图片，经试验效果不错

classes_unwanted = [    #不想要的数据种类
    'car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','horse','sheep','cow','elephant',
    'bear','zebra','giraffe','frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
]
indoorCategories = [{"supercategory": "person", "id": 1, "name": "person"}, 
                    {"supercategory": "vehicle", "id": 2, "name": "bicycle"}, 
                    {"supercategory": "animal", "id": 17, "name": "cat"}, 
                    {"supercategory": "animal", "id": 18, "name": "dog"}, 
                    {"supercategory": "accessory", "id": 27, "name": "backpack"}, 
                    {"supercategory": "accessory", "id": 28, "name": "umbrella"}, 
                    {"supercategory": "accessory", "id": 31, "name": "handbag"}, 
                    {"supercategory": "accessory", "id": 32, "name": "tie"}, 
                    {"supercategory": "accessory", "id": 33, "name": "suitcase"}, 
                    {"supercategory": "kitchen", "id": 44, "name": "bottle"}, 
                    {"supercategory": "kitchen", "id": 46, "name": "wine glass"}, 
                    {"supercategory": "kitchen", "id": 47, "name": "cup"}, 
                    {"supercategory": "kitchen", "id": 48, "name": "fork"}, 
                    {"supercategory": "kitchen", "id": 49, "name": "knife"}, 
                    {"supercategory": "kitchen", "id": 50, "name": "spoon"}, 
                    {"supercategory": "kitchen", "id": 51, "name": "bowl"}, 
                    {"supercategory": "food", "id": 52, "name": "banana"}, 
                    {"supercategory": "food", "id": 53, "name": "apple"}, 
                    {"supercategory": "food", "id": 54, "name": "sandwich"}, 
                    {"supercategory": "food", "id": 55, "name": "orange"}, 
                    {"supercategory": "food", "id": 56, "name": "broccoli"}, 
                    {"supercategory": "food", "id": 57, "name": "carrot"}, 
                    {"supercategory": "food", "id": 58, "name": "hot dog"}, 
                    {"supercategory": "food", "id": 59, "name": "pizza"}, 
                    {"supercategory": "food", "id": 60, "name": "donut"}, 
                    {"supercategory": "food", "id": 61, "name": "cake"}, 
                    {"supercategory": "furniture", "id": 62, "name": "chair"}, 
                    {"supercategory": "furniture", "id": 63, "name": "couch"}, 
                    {"supercategory": "furniture", "id": 64, "name": "potted plant"}, 
                    {"supercategory": "furniture", "id": 65, "name": "bed"}, 
                    {"supercategory": "furniture", "id": 67, "name": "dining table"}, 
                    {"supercategory": "furniture", "id": 70, "name": "toilet"}, 
                    {"supercategory": "electronic", "id": 72, "name": "tv"}, 
                    {"supercategory": "electronic", "id": 73, "name": "laptop"}, 
                    {"supercategory": "electronic", "id": 74, "name": "mouse"}, 
                    {"supercategory": "electronic", "id": 75, "name": "remote"}, 
                    {"supercategory": "electronic", "id": 76, "name": "keyboard"}, 
                    {"supercategory": "electronic", "id": 77, "name": "cell phone"}, 
                    {"supercategory": "appliance", "id": 78, "name": "microwave"}, 
                    {"supercategory": "appliance", "id": 79, "name": "oven"}, 
                    {"supercategory": "appliance", "id": 80, "name": "toaster"}, 
                    {"supercategory": "appliance", "id": 81, "name": "sink"}, 
                    {"supercategory": "appliance", "id": 82, "name": "refrigerator"}, 
                    {"supercategory": "indoor", "id": 84, "name": "book"}, 
                    {"supercategory": "indoor", "id": 85, "name": "clock"}, 
                    {"supercategory": "indoor", "id": 86, "name": "vase"}, 
                    {"supercategory": "indoor", "id": 87, "name": "scissors"}, 
                    {"supercategory": "indoor", "id": 88, "name": "teddy bear"}, 
                    {"supercategory": "indoor", "id": 89, "name": "hair drier"}, 
                    {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]
# COCO数据集的11个大类80个小类
# 'person','backpack','umbrella','handbag','tie','suitcase',                                     #person&accessory
# 'bicycle','car','motorcycle','airplane','bus','train','truck','boat',                          #vehicle
# 'traffic light','fire hydrant','stop sign','parking meter','bench',                            #outdoor object
# 'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',                  #animal
# 'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
# 'skateboard','surfboard','tennis racket',                                                      #sport
# 'bottle','wine glass','cup','fork','knife','spoon','bowl',                                     #kirchenware
# 'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',     #food
# 'chair','couch','potted plant','bed','dining table','toilet',                                  #furniture
# 'tv','laptop','mouse','remote','keyboard','cell phone',                                        #electronics
# 'microwave','oven','toaster','sink','refrigerator',                                            #appliance
# 'book','clock','vase','scissors','teddy bear','hair drier','toothbrush',                       #indoor object

def cocoget(datasets):
    json_path = '{}annotations/instances_{}.json'.format(dataDir, datasets)
    json_info = json.load(open(json_path,'r'))

    coco=COCO(json_path)

    imgIds=coco.getImgIds()     #所有图片id

    ucatIds = coco.getCatIds(catNms=classes_unwanted)   #不需要的类别

    uimgIds = []        #不需要的图片id
    for ucatId in ucatIds:
        uimgIds.append(coco.getImgIds(catIds=ucatId))

    uimgIds = [item for sublist in uimgIds for item in sublist]   #id展平
    uimgIds  = list(set(uimgIds))                                 #去除重复id

    for id in uimgIds:
        imgIds.remove(id)                   #得到剩下想要的imgIds
    
    mkr(r'{}/{}'.format(savepath, datasets))    #创建保存路径
    
    #   只需要有原COCO数据集搭配json文件即可，无需移动图片
    # for id in imgIds:
    #     imgname = "%012d" % id
    #     img_path = r'{}{}/{}.jpg'.format(dataDir, datasets,imgname)
    #     save_path = r'{}{}/'.format(savepath, datasets)
    #     shutil.copy(img_path,save_path)             #复制到保存路径中
    #     print(img_path)


    images = coco.loadImgs(ids=imgIds)          
    annIds = coco.getAnnIds(imgIds=imgIds)      
    annotations = coco.loadAnns(ids=annIds)     
    categories = json_info['categories']

    my_coco = {     #创建字典
        "images":images,
        "annotations":annotations,
        "categories":indoorCategories,
    }
    my_coco_json = json.dumps(my_coco)
    f = open(os.path.join(anno_dir+'/instances_{}.json'.format(datasets)), 'w') #保存对应json
    f.write(my_coco_json)
    f.close()

def mkr(path):
    if os.path.exists(path):
        # shutil.rmtree(path)
        # os.mkdir(path)
        pass
    else:
        os.mkdir(path)

if __name__ == "__main__":
    mkr(r'{}/annotations'.format(savepath))
    for datasets in datasets_list:
        cocoget(datasets)
    print('数据集提取完成')

