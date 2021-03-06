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
"""
DATA = "3d_reference_thick1"
DATA_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/metal_structure/Kia_metal_structure_v1"

TRAIN_DATA = "Kia_metal_structure_train"
TRAIN_DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_train_cocoformat.json")
TRAIN_DATA_IMG_PATH = os.path.join(DATA_PATH, "train")

TEST_DATA = "Kia_metal_structure_test"
TEST_DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_test_cocoformat.json")
TEST_DATA_IMG_PATH = os.path.join(DATA_PATH, "test")

CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
"""
DATA = "3d_reference_thick1"
DATA_PATH = "/home/jinsuby/Desktop/PycharmProjects/data/Kia/3d_reference/Kia_coco_thick_1_v3"

TRAIN_DATA = "Kia_3d_reference_train"
TRAIN_DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_train_cocoformat.json")
TRAIN_DATA_IMG_PATH = os.path.join(DATA_PATH, "train")

TEST_DATA = "Kia_3d_reference_test"
TEST_DATA_JSON_PATH = os.path.join(DATA_PATH, "annotations/Kia_test_cocoformat.json")
TEST_DATA_IMG_PATH = os.path.join(DATA_PATH, "test")

CONFIG_FILE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
register_coco_instances(TRAIN_DATA, {}, TRAIN_DATA_JSON_PATH, TRAIN_DATA_IMG_PATH)
register_coco_instances(TEST_DATA, {}, TEST_DATA_JSON_PATH, TEST_DATA_IMG_PATH)

Kia_trainval_metadata = MetadataCatalog.get(TRAIN_DATA)
Kia_trainval_dataset_dicts = DatasetCatalog.get(TRAIN_DATA)

Kia_test_metadata = MetadataCatalog.get(TEST_DATA)
Kia_test_dataset_dicts = DatasetCatalog.get(TEST_DATA)

#Train!
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
print("#######################cfg#########################")
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
cfg.DATASETS.TRAIN = TRAIN_DATA
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE)  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Numbers of classes
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough, but you can certainly train longer
cfg.OUTPUT_DIR = "./output/" + DATA + "/" + CONFIG_FILE.replace("/", "-").replace(".yaml", "") + "-eps" + str(cfg.SOLVER.MAX_ITER) + "-thresh" + str(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST) + "-bat" + str(cfg.SOLVER.IMS_PER_BATCH) + "/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#cfg.MODEL.DEVICE = "cpu" # If you want, can you use only "cpu".

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

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(TEST_DATA, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, TEST_DATA)
inference_on_dataset(trainer.model, val_loader, evaluator)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(Kia_test_dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print("outputs : ", outputs)
    print("outputs[instances] : ", outputs["instances"])
    v = Visualizer(im[:, :, ::-1],
                   metadata=Kia_test_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #out = v.draw_dataset_dict(d) # draw predictions with using Annotations

    cv2.imwrite(cfg.OUTPUT_DIR + "%s" %(d["file_name"].split("/")[-1]), out.get_image()[:, :, ::-1])