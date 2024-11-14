#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
# 可视化！！
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.visualizer import Visualizer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from contextlib import ExitStack
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
# from detectron2.data.datasets import register_pascal_voc, register_ACDC_instances, register_cityscape
import cv2

logger = logging.getLogger("detectron2")

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, allow_cached_coco=False))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            out = model(data)
            print(out)
            break


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    args.config_file = "/public/home/caoshilei/test/detectron2/tools/cfg.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def adaptAndEval(model, data_loader, evaluator):
    evaluator.reset()
    for iter, data in enumerate(data_loader):
        outputs = model(data)
        evaluator.process(data, outputs)

    return evaluator.evaluate()


def main(args):
    cfg = setup(args)
    register_cityscape("Train", "/public/home/caoshilei/test/detectron2/datasets/cityscapes", "train_s")
    register_cityscape("Test", "/public/home/caoshilei/test/detectron2/datasets/cityscapes", "test_s")

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        model.eval()
    # data_loader = build_detection_train_loader(cfg)
    dataSet = cfg.DATASETS.TEST[0]
    data_loader = build_detection_test_loader(cfg, dataSet)

    for iter, data in enumerate(data_loader):
        outputs = model(data)
        image = cv2.imread(data[0]["file_name"])
        v1 = Visualizer(image[:, :, ::-1], MetadataCatalog.get(dataSet), scale=1.2)
        out = v1.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])
        dataDict = [d for d in DatasetCatalog.get(dataSet) if d["file_name"] == data[0]["file_name"]][0]
        v2 = Visualizer(image[:, :, ::-1], MetadataCatalog.get(dataSet), scale=1.2)
        gt = v2.draw_dataset_dict(dataDict)
        cv2.imwrite("gt.jpg", gt.get_image()[:, :, ::-1])

    # data_loaders = [build_detection_test_loader(cfg, datasetName) for datasetName in cfg.DATASETS.TEST]

    # evaluators = [get_evaluator(cfg, datasetName, os.path.join(cfg.OUTPUT_DIR, "inference", datasetName)) for datasetName in cfg.DATASETS.TEST]
    # # evaluator.reset()
    # with ExitStack() as stack:
    #     stack.enter_context(torch.no_grad())
    #     for epoch in range(1):
    #         print("epoch:", epoch)
    #         results = OrderedDict()
    #         for iter, data_loader, evaluator in zip(range(len(data_loaders)), data_loaders, evaluators):
    #             print("dataSet:", iter)
    #             evaluator.reset()
    #             result = adaptAndEval(model, data_loader, evaluator)
    #             results[iter] = result
    #             break
    #         print(results)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
