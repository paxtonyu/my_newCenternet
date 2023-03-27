# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from centernet.config import add_centernet_config
# constants
WINDOW_NAME = "CenterNet2 detections"

from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
        cfg.MODEL.CENTERNET.INFERENCE_TH = args.confidence_threshold
        cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

#detectron2的默认参数可在detectron2/config/defaults.py中查看
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser  

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)    #detectron的multiprocessing
    #mp是python的多进程模块，set_start_method()是设置启动方式，spawn是新建进程，force=True是强制使用spawn
    args = get_parser().parse_args()
    #上面这话是将parser.py中的get_parser()函数返回的parser对象赋值给args,args是一个对象，里面包含了所有的参数
    logger = setup_logger()
    #setup_logger()是detectron2.utils.logger中的函数，用于设置日志
    logger.info("Arguments: " + str(args))
    #logger.info()是logging模块中的函数，用于打印日志，这里打印的是参数
    cfg = setup_cfg(args)
    #setup_cfg()是上面定义的函数，用于设置配置文件，cfg里具体包含的内容可以看detectron2.config.config.py
    demo = VisualizationDemo(cfg)
    #VisualizationDemo()是一个类，定义在predictor.py中，用于可视化
    output_file = None
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            #glob模块是用于查找符合特定规则的文件路径名，os.path.expanduser()是os.path模块中的函数，用于展开用户路径
            files = os.listdir(args.input[0])
            #listdir()是os模块中的函数，用于返回指定的文件夹包含的文件或文件夹的名字的列表
            #输入的是一个文件夹的路径，返回的是一个列表，里面是文件夹中的文件名，如图片名称列表
            args.input = [args.input[0] + x for x in files]
            #将文件夹路径和文件名拼接起来，得到图片的路径
            assert args.input, "The input path(s) was not found"
            #assert是python中的断言语句，如果assert后面的条件为False，则会抛出AssertionError异常
            #如果args.input为空,则抛出异常
        visualizer = VideoVisualizer(
            MetadataCatalog.get(
            # 这里的MetadataCatalog是detectron2.data.catalog中的类,MetadataCatalog.get()是其中的一个函数,
            # 输入的是cfg.DATASETS.TEST[0]，返回的是一个字典
                cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
            # cfg.DATASETS.TEST[0]是一个字符串，是数据集的名字，比如coco_2017_train
            ), 
            instance_mode=ColorMode.IMAGE)
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # tqdm是一个进度条模块，tqdm.tqdm()是其中的一个函数，用于显示进度条
            # args.input是一个列表，里面是图片的路径
            # disable=not args.output是一个布尔值，如果args.output为None，则disable为True，否则为False

            # use PIL, to be consistent with evaluation

            img = read_image(path, format="BGR")
            # read_image()是detectron2.utils.image中的函数，用于读取图片，format="BGR"表示以BGR的格式读取
            start_time = time.time()
            # time.time()是time模块中的函数，用于返回当前时间的时间戳
            predictions, visualized_output = demo.run_on_image(
                img, visualizer=visualizer)
            # run_on_image()是VisualizationDemo类中的一个函数，用于预测图片
            # img是图片，visualizer是一个VideoVisualizer类的对象，在上文中定义过
            # 返回的是predictions(ditc)和vis_output(VisImage)
            if 'instances' in predictions: #如果predictions中有instances，prediction是一个字典，里面包含了instances以及proposals以及其他的信息
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )
            else:
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["proposals"]), time.time() - start_time
                    )
                )

            if args.output:
                if os.path.isdir(args.output):#如果args.output是一个文件夹
                    assert os.path.isdir(args.output), args.output#断言args.output是一个文件夹
                    out_filename = os.path.join(args.output, os.path.basename(path))#将args.output和path拼接起来，得到输出文件的路径
                    #os.path.basename()返回的是path的最后一个文件名
                    visualized_output.save(out_filename)#将可视化的结果保存到out_filename中
                    #visualized_output是一个VisImage类的对象
                    #save()是VisImage类中的一个函数，用于保存可视化的结果
                else:
                    # assert len(args.input) == 1, "Please specify a directory with args.output"
                    # out_filename = args.output
                    if output_file is None:
                        width = visualized_output.get_image().shape[1]
                        height = visualized_output.get_image().shape[0]
                        frames_per_second = 15
                        output_file = cv2.VideoWriter(
                            filename=args.output,
                            # some installation of opencv may not support x264 (due to its license),
                            # you can try other format (e.g. MPEG)
                            fourcc=cv2.VideoWriter_fourcc(*"x264"),
                            fps=float(frames_per_second),
                            frameSize=(width, height),
                            isColor=True,
                        )
                    output_file.write(visualized_output.get_image()[:, :, ::-1])
            else:
                # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(1 ) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = 15 # video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            # assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)

            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            cv2.imshow(basename, vis_frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
