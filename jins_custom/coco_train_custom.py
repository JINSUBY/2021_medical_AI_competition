# Some basic setup:
# Setup detectron2 logger
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

register_coco_instances("custom_coco_train", {}, "/home/jinsuby/Desktop/PycharmProjects/data/coco/annotations/instances_train2017.json", "/home/jinsuby/Desktop/PycharmProjects/data/coco/train2017")
register_coco_instances("custom_coco_val", {}, "/home/jinsuby/Desktop/PycharmProjects/data/coco/annotations/instances_val2017.json", "/home/jinsuby/Desktop/PycharmProjects/data/coco/val2017")

coco_train_metadata = MetadataCatalog.get("custom_coco_train")
coco_train_dataset_dicts = DatasetCatalog.get("custom_coco_train")

#Train!
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("custom_coco_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Numbers of classes

#cfg.MODEL.DEVICE = "cpu" # If you want, can you use only "cpu".

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
class CocoTrainer(DefaultTrainer):
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
                os.makedirs("coco_eval", exist_ok=True)
                output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("custom_coco_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "custom_coco_val")
inference_on_dataset(trainer.model, val_loader, evaluator)