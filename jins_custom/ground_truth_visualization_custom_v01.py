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

DATA = "custom_coco_val"
DATA_JSON_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/Kia_coco_thick_4_v2/annotations/Kia_test_cocoformat.json"
DATA_IMG_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/Kia_coco_thick_4_v2/test"
CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
MODEL_FILE_PATH = "Kia-Detection_faster_rcnn_R_50_FPN_1x-eps1000_v2/model_final.pth"

register_coco_instances(DATA, {}, DATA_JSON_PATH, DATA_IMG_PATH)

DATA_metadata = MetadataCatalog.get(DATA)
DATA_dataset_dicts = DatasetCatalog.get(DATA)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Numbers of classes
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, MODEL_FILE_PATH)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom testing threshold
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
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu")) # draw instances with using predictors of model
    out = v.draw_dataset_dict(d) # draw instances with using Annotations(ground truth)
    print(out.get_image().shape)
    print(out.get_image()[:, :, ::-1].shape)
    cv2.imwrite("./output/%s" %(d["file_name"].split("/")[-1]), out.get_image()[:, :, ::-1])