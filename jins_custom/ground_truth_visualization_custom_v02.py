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

DATA = "Kia_metal_structure_test"
DATA_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/metal_structure/Kia_metal_structure_v1"

DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_test_cocoformat.json")
DATA_IMG_PATH = os.path.join(DATA_PATH, "test")

CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
MODEL_FILE_PATH = "Kia_metal_structure-COCO-Detection-faster_rcnn_R_50_FPN_1x-eps1000-thresh0.5-bat16/model_final.pth"
"""

DATA = "Kia_3d_reference_test"
DATA_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/3d_reference/Kia_coco_thick_1_v3"

DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_test_cocoformat.json")
DATA_IMG_PATH = os.path.join(DATA_PATH, "test")

CONFIG_FILE = "COCO-Detection/rpn_R_50_FPN_1x.yaml"
MODEL_FILE_PATH = "Kia_thick1-COCO-Detection-rpn_R_50_FPN_1x-eps3000-thresh0.5-bat16/model_final.pth"
"""
register_coco_instances(DATA, {}, DATA_JSON_PATH, DATA_IMG_PATH)

DATA_metadata = MetadataCatalog.get(DATA)
DATA_dataset_dicts = DatasetCatalog.get(DATA)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Numbers of classes
cfg.MODEL.WEIGHTS = os.path.join("./output/", MODEL_FILE_PATH)  # path to the model we just trained
cfg.OUTPUT_DIR = "./output/" + MODEL_FILE_PATH.split("/")[0] + "/ground_truth_visualization/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(DATA_dataset_dicts, 20):
    im = cv2.imread(d["file_name"])
    print(im.shape)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print(outputs)
    print(outputs["instances"])
    v = Visualizer(im[:, :, ::-1],
                   metadata=DATA_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu")) # draw instances with using predictors of model
    out = v.draw_dataset_dict(d) # draw instances with using Annotations(ground truth)
    print(out.get_image().shape)
    print(out.get_image()[:, :, ::-1].shape)
    cv2.imwrite(cfg.OUTPUT_DIR + "%s" %(d["file_name"].split("/")[-1]), out.get_image()[:, :, ::-1])