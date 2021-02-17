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

TEST_DATA = "Kia_region_coco_test"
TEST_DATA_JSON_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/Kia_coco_thick_4/Kia_test_cocoformat.json"
TEST_DATA_IMG_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/Kia_coco_thick_4/JPEGimages"
CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
MODEL_FILE_PATH = "Kia-Detection_faster_rcnn_R_50_FPN_1x-eps1000/model_final.pth"

register_coco_instances(TEST_DATA, {}, TEST_DATA_JSON_PATH, TEST_DATA_IMG_PATH)

Kia_test_metadata = MetadataCatalog.get(TEST_DATA)
Kia_test_dataset_dicts = DatasetCatalog.get(TEST_DATA)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
cfg.DATASETS.TEST = TEST_DATA
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Numbers of classes
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, MODEL_FILE_PATH)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(Kia_test_dataset_dicts, 20):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print("outputs : ", outputs)
    print("outputs[instances] : ", outputs["instances"])
    v = Visualizer(im[:, :, ::-1],
                   metadata=Kia_test_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #out = v.draw_dataset_dict(d) # draw predictions with using Annotations
    print("out.get_image().shape : ", out.get_image().shape)
    print("out.get_image()[:, :, ::-1].shape : ", out.get_image()[:, :, ::-1].shape)

    cv2.imwrite("./output/%s" %(d["file_name"].split("/")[-1]), out.get_image()[:, :, ::-1])


"""
img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
basename = os.path.basename(dic["file_name"])

predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
vis = Visualizer(img, metadata)
vis_pred = vis.draw_instance_predictions(predictions).get_image()

vis = Visualizer(img, metadata)
vis_gt = vis.draw_dataset_dict(dic).get_image()

concat = np.concatenate((vis_pred, vis_gt), axis=1)
cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
"""