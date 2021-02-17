import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances


"""
DATA_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/metal_structure/Kia_metal_structure_v1"

DATA = "metal_structure"
DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_test_cocoformat.json")
DATA_IMG_PATH = os.path.join(DATA_PATH, "test")

CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
MODEL_FILE_PATH = "COCO-Detection-faster_rcnn_R_50_FPN_1x-eps3000-thresh0.5/model_final.pth"
"""
DATA_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/3d_reference/Kia_coco_thick_1_v3"

DATA = "3d_reference_thick1"
DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_test_cocoformat.json")
DATA_IMG_PATH = os.path.join(DATA_PATH, "test")

CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
MODEL_FILE_PATH = "COCO-Detection-faster_rcnn_R_50_FPN_1x-eps3000-thresh0.5/model_final.pth"

register_coco_instances(DATA, {}, DATA_JSON_PATH, DATA_IMG_PATH)

DATA_metadata = MetadataCatalog.get(DATA)
DATA_dataset_dicts = DatasetCatalog.get(DATA)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Numbers of classes
cfg.MODEL.WEIGHTS = os.path.join("./output/", DATA, MODEL_FILE_PATH)   # path to the model we just trained
cfg.OUTPUT_DIR = os.path.join("./output", DATA, MODEL_FILE_PATH.split("/")[0], "reference_pixel_extraction")
print(cfg.OUTPUT_DIR)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
predictor = DefaultPredictor(cfg)

#color 설정
blue_color = (255,0,0)
green_color = (0,255,0)
red_color = (0,0,255)
white_color = (255,255,255)
black_color = (0,0,0)
from detectron2.utils.visualizer import ColorMode

for d in random.sample(DATA_dataset_dicts, 20):
    im = cv2.imread(d["file_name"])
    f = open(os.path.join(os.path.dirname(cfg.OUTPUT_DIR), "visualization-thresh0.5", d["file_name"].split("/")[-1].replace(".jpg", ".txt")), "r")
    data = f.readlines()
    for i in data:
        i = i.split("\t")
        if i[0] == "8":
            #im[i[1], i[2]] = [0, 255, 0]
            im = cv2.line(im, (int(i[1]), int(i[2])), (int(i[1]), int(i[2])), red_color, 5)
    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, "%s" % (d["file_name"].split("/")[-1])), im)