# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse ACDC-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = [
    "load_ACDC_json",
]

def _get_name2id_dict():
    openset_name2id={
        'person':0,
        'car':1,
        'train':2,
        'rider':3,
        'truck':4,
        'motorcycle':5,
        'bicycle':6,
        'bus':7,
        'unknown':8
    }
    return openset_name2id

def get_openset_settings_cityscapes(setting):

    NMES = ['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus']
    UNK = ["unknown"]

    if setting == 1:  # different semantics
        BASE_CLASSES = ['car', 'truck', 'bus']
        NOVEL_CLASSES = ['person', 'motorcycle', 'train', 'bicycle', 'rider']
    elif setting == 2:  # similar semantics
        BASE_CLASSES = ['person', 'bicycle', 'bus']
        NOVEL_CLASSES = ['car', 'truck', 'train', 'motorcycle', 'rider']
    elif setting == 3:  # frequency down
        BASE_CLASSES = ['person', 'car', 'rider']
        NOVEL_CLASSES = ['bicycle', 'train', 'truck', 'motorcycle', 'bus']
    elif setting == 4:  # frequency top
        BASE_CLASSES = ['motorcycle', 'truck', 'bus']
        NOVEL_CLASSES = ['person', 'train', 'car', 'bicycle', 'rider']

    ALL_CLASSES = tuple(BASE_CLASSES+NOVEL_CLASSES)
    CLASS_NAMES = tuple(BASE_CLASSES+UNK)

    return BASE_CLASSES, NOVEL_CLASSES, ALL_CLASSES, CLASS_NAMES

def get_openset_cityscapes_class(openset_setting, is_train):
    BASE_CLASSES, NOVEL_CLASSES, ALL_CLASSES, CLASS_NAMES = get_openset_settings_cityscapes(openset_setting)
    if is_train:
        class_name = BASE_CLASSES
    else:
        class_name = CLASS_NAMES
    return class_name

def load_ACDC_json(json_file, image_root, openset_setting=1, dataset_name=None, is_train=False):
    import pdb
    pdb.set_trace()
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    # 重新对数据集的category_id进行编号,ACDC数据集使用cityscapes相同的labels进行编码
    from cityscapesscripts.helpers.labels import id2label, name2label

    BASE_CLASSES, NOVEL_CLASSES, ALL_CLASSES, CLASS_NAMES = get_openset_settings_cityscapes(openset_setting)

    # 得到的索引一定要连续
    class_name=get_openset_cityscapes_class(openset_setting,is_train)
    labels = [_get_name2id_dict()[l] for l in class_name]
    dataset_id_to_contiguous_id = {l: idx for idx, l in enumerate(labels)}
    id_map = dataset_id_to_contiguous_id

    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS

            label = id2label[int(obj['category_id'])]
            if is_train:
                if label.name in BASE_CLASSES:
                    obj['category_id'] = _get_name2id_dict()[label.name]
            else:
                if label.name in BASE_CLASSES:
                    obj['category_id'] = _get_name2id_dict()[label.name]
                else:
                    obj['category_id'] = _get_name2id_dict()['unknown']

            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
              "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


# def convert_to_coco_dict(dataset_name):
#     """
#     Convert an instance detection/segmentation or keypoint detection dataset
#     in detectron2's standard format into COCO json format.
#
#     Generic dataset description can be found here:
#     https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset
#
#     COCO data format description can be found here:
#     http://cocodataset.org/#format-data
#
#     Args:
#         dataset_name (str):
#             name of the source dataset
#             Must be registered in DatastCatalog and in detectron2's standard format.
#             Must have corresponding metadata "thing_classes"
#     Returns:
#         coco_dict: serializable dict in COCO json format
#     """
#
#     dataset_dicts = DatasetCatalog.get(dataset_name)
#     metadata = MetadataCatalog.get(dataset_name)
#
#     # unmap the category mapping ids for COCO
#     if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
#         reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
#         reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
#     else:
#         reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa
#
#     categories = [
#         {"id": reverse_id_mapper(id), "name": name}
#         for id, name in enumerate(metadata.thing_classes)
#     ]
#
#     logger.info("Converting dataset dicts into COCO format")
#     coco_images = []
#     coco_annotations = []
#
#     for image_id, image_dict in enumerate(dataset_dicts):
#         coco_image = {
#             "id": image_dict.get("image_id", image_id),
#             "width": int(image_dict["width"]),
#             "height": int(image_dict["height"]),
#             "file_name": str(image_dict["file_name"]),
#         }
#         coco_images.append(coco_image)
#
#         anns_per_image = image_dict.get("annotations", [])
#         for annotation in anns_per_image:
#             # create a new dict with only COCO fields
#             coco_annotation = {}
#
#             # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
#             bbox = annotation["bbox"]
#             if isinstance(bbox, np.ndarray):
#                 if bbox.ndim != 1:
#                     raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
#                 bbox = bbox.tolist()
#             if len(bbox) not in [4, 5]:
#                 raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
#             from_bbox_mode = annotation["bbox_mode"]
#             to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
#             bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)
#
#             # COCO requirement: instance area
#             if "segmentation" in annotation:
#                 # Computing areas for instances by counting the pixels
#                 segmentation = annotation["segmentation"]
#                 # TODO: check segmentation type: RLE, BinaryMask or Polygon
#                 if isinstance(segmentation, list):
#                     polygons = PolygonMasks([segmentation])
#                     area = polygons.area()[0].item()
#                 elif isinstance(segmentation, dict):  # RLE
#                     area = mask_util.area(segmentation).item()
#                 else:
#                     raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
#             else:
#                 # Computing areas using bounding boxes
#                 if to_bbox_mode == BoxMode.XYWH_ABS:
#                     bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
#                     area = Boxes([bbox_xy]).area()[0].item()
#                 else:
#                     area = RotatedBoxes([bbox]).area()[0].item()
#
#             if "keypoints" in annotation:
#                 keypoints = annotation["keypoints"]  # list[int]
#                 for idx, v in enumerate(keypoints):
#                     if idx % 3 != 2:
#                         # COCO's segmentation coordinates are floating points in [0, H or W],
#                         # but keypoint coordinates are integers in [0, H-1 or W-1]
#                         # For COCO format consistency we substract 0.5
#                         # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
#                         keypoints[idx] = v - 0.5
#                 if "num_keypoints" in annotation:
#                     num_keypoints = annotation["num_keypoints"]
#                 else:
#                     num_keypoints = sum(kp > 0 for kp in keypoints[2::3])
#
#             # COCO requirement:
#             #   linking annotations to images
#             #   "id" field must start with 1
#             coco_annotation["id"] = len(coco_annotations) + 1
#             coco_annotation["image_id"] = coco_image["id"]
#             coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
#             coco_annotation["area"] = float(area)
#             coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
#             coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))
#
#             # Add optional fields
#             if "keypoints" in annotation:
#                 coco_annotation["keypoints"] = keypoints
#                 coco_annotation["num_keypoints"] = num_keypoints
#
#             if "segmentation" in annotation:
#                 seg = coco_annotation["segmentation"] = annotation["segmentation"]
#                 if isinstance(seg, dict):  # RLE
#                     counts = seg["counts"]
#                     if not isinstance(counts, str):
#                         # make it json-serializable
#                         seg["counts"] = counts.decode("ascii")
#
#             coco_annotations.append(coco_annotation)
#
#     logger.info(
#         "Conversion finished, "
#         f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
#     )
#
#     info = {
#         "date_created": str(datetime.datetime.now()),
#         "description": "Automatically generated COCO json file for Detectron2.",
#     }
#     coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
#     if len(coco_annotations) > 0:
#         coco_dict["annotations"] = coco_annotations
#     return coco_dict
#
#
# def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
#     """
#     Converts dataset into COCO format and saves it to a json file.
#     dataset_name must be registered in DatasetCatalog and in detectron2's standard format.
#
#     Args:
#         dataset_name:
#             reference from the config file to the catalogs
#             must be registered in DatasetCatalog and in detectron2's standard format
#         output_file: path of json file that will be saved to
#         allow_cached: if json file is already present then skip conversion
#     """
#
#     # TODO: The dataset or the conversion script *may* change,
#     # a checksum would be useful for validating the cached data
#
#     PathManager.mkdirs(os.path.dirname(output_file))
#     with file_lock(output_file):
#         if PathManager.exists(output_file) and allow_cached:
#             logger.warning(
#                 f"Using previously cached COCO format annotations at '{output_file}'. "
#                 "You need to clear the cache file if your dataset has been modified."
#             )
#         else:
#             logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
#             coco_dict = convert_to_coco_dict(dataset_name)
#
#             logger.info(f"Caching COCO format annotations at '{output_file}' ...")
#             tmp_file = output_file + ".tmp"
#             with PathManager.open(tmp_file, "w") as f:
#                 json.dump(coco_dict, f)
#             shutil.move(tmp_file, output_file)
#
#
# def register_ACDC_instances(name, json_file, image_root):
#     # todo: 将此函数放在builtin中用于处理不同的openset类型
#     assert isinstance(name, str), name
#     assert isinstance(json_file, (str, os.PathLike)), json_file
#     assert isinstance(image_root, (str, os.PathLike)), image_root
#     # 1. register a function which returns dicts
#     DatasetCatalog.register(name, lambda: load_ACDC_json(json_file, image_root, name))
#
#     # 2. Optionally, add metadata about this dataset,
#     # since they might be useful in evaluation, visualization or logging
#     MetadataCatalog.get(name).set(
#         thing_classes=list(CLASS_NAMES), json_file=json_file, image_root=image_root, evaluator_type="coco"
#     )
#
#
# if __name__ == "__main__":
#     """
#     Test the ACDC json dataset loader.
#
#     Usage:
#         python -m detectron2.data.datasets.coco \
#             path/to/json path/to/image_root
#
#     """
#     from detectron2.utils.logger import setup_logger
#     from detectron2.utils.visualizer import Visualizer
#     import detectron2.data.datasets  # noqa # add pre-defined metadata
#     import sys
#
#     logger = setup_logger(name=__name__)
#
#     dicts = load_ACDC_json(sys.argv[1], sys.argv[2])
#     logger.info("Done loading {} samples.".format(len(dicts)))
#
#     register_ACDC_instances("ACDC", sys.argv[1], sys.argv[2])
#
#     dirname = "coco-data-vis"
#     os.makedirs(dirname, exist_ok=True)
#     for d in dicts:
#         print(d)
#         img = np.array(Image.open(d["file_name"]))
#         visualizer = Visualizer(img, metadata=MetadataCatalog.get("ACDC"))
#         vis = visualizer.draw_dataset_dict(d)
#         fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
#         vis.save(fpath)