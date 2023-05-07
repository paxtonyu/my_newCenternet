from pycocotools.coco import COCO
import os
import json

# 需要设置的路径
dataDir = './datasets/coco/'             #原始数据集路径
savepath = "./datasets/coco_my/"         #保存数据集路径
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
    print('读取{}中...'.format(datasets))
    coco=COCO(json_path)
    print('json读取完成,开始提取文件标注...')
    imgIds=coco.getImgIds()     #所有图片id

    ucatIds = coco.getCatIds(catNms=classes_unwanted)   #不需要的类别

    uimgIds = []        #不需要的图片id
    for ucatId in ucatIds:
        uimgIds.append(coco.getImgIds(catIds=ucatId))

    uimgIds = [item for sublist in uimgIds for item in sublist]   #id展平
    uimgIds  = list(set(uimgIds))                                 #去除重复id

    for id in uimgIds:
        imgIds.remove(id)                   #得到剩下想要的imgIds
    
    mkr(r'{}{}'.format(savepath, datasets))    #创建保存路径
    
    #   只需要有原COCO数据集搭配json文件即可，无需移动图片
    # for id in imgIds:
    #     imgname = "%012d" % id
    #     img_path = r'{}{}/{}.jpg'.format(dataDir, datasets,imgname)
    #     save_path = r'{}{}/'.format(savepath, datasets)
    #     shutil.copy(img_path,save_path)             #复制到保存路径中
    #     print(img_path)

    print('开始保存json文件...')
    images = coco.loadImgs(ids=imgIds)          
    annIds = coco.getAnnIds(imgIds=imgIds)      
    annotations = coco.loadAnns(ids=annIds)
    categories = []    
    for cat in json_info['categories']:
        if cat['id'] not in ucatIds:
            categories.append(cat)


    my_coco = {     #创建字典
        "images":images,
        "annotations":annotations,
        "categories":categories,
    }
    my_coco_json = json.dumps(my_coco)
    f = open(os.path.join(anno_dir+'my_instances_{}.json'.format(datasets)), 'w') #保存对应json
    f.write(my_coco_json)
    f.close()
    print('{}保存完成'.format(datasets))

def mkr(path):
    if os.path.exists(path):
        # shutil.rmtree(path)
        # os.mkdir(path)
        pass
    else:
        os.makedirs(path)       #单级目录使用mkdir，多级目录使用makedirs

if __name__ == "__main__":
    mkr(r'{}annotations'.format(savepath))
    for datasets in datasets_list:
        cocoget(datasets)
    print('所有数据集均提取完成(*^_^*)')

