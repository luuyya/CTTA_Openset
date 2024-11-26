# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)

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

def get_openset_cityscapes_class(openset_setting, is_train):
    BASE_CLASSES, NOVEL_CLASSES, ALL_CLASSES, CLASS_NAMES = get_openset_settings_cityscapes(openset_setting)
    if is_train:
        class_name = BASE_CLASSES
    else:
        class_name = CLASS_NAMES
    return class_name

def _get_cityscapes_files(image_dir, gt_dir):
    """

    Args:
        image_dir: 像素照片地址
        gt_dir: 标注文件地址

    Returns:
        一个列表，表中将 图片、instance、label和ploygons匹配存储

    """
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_cityscapes_instances_openset(image_dir, gt_dir, openset_setting=1, from_json=True, to_polygons=True, is_train=True):
    """

    Args:
        image_dir: 像素照片地址
        gt_dir: 标注文件地址
        from_json:
        to_polygons:

    Returns:

    """
    # 基本一样，应该可以运行
    if from_json:
        assert to_polygons, (
            "Cityscapes's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
        )
    files = _get_cityscapes_files(image_dir, gt_dir)

    logger.info("Preprocessing cityscapes annotations in open-set setting {}...".format(openset_setting))
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map( #多了两个参数输入
        functools.partial(_cityscapes_files_to_dict, openset_setting=openset_setting, from_json=from_json, to_polygons=to_polygons, is_train=is_train),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))
    print(is_train)
    print('+'*100)

    # 得到的索引一定要连续
    class_name=get_openset_cityscapes_class(openset_setting,is_train)
    labels = [_get_name2id_dict()[l] for l in class_name]
    dataset_id_to_contiguous_id = {l: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret


def _cityscapes_files_to_dict(files, openset_setting, from_json, to_polygons, is_train):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    # 多了两个判断语句，应该不影响
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, json_file = files

    annos = []

    if from_json:
        from shapely.geometry import MultiPolygon, Polygon

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }

        polygons_union = Polygon()

        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

            poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)

            if not label.hasInstances or label.ignoreInEval:
                polygons_union = polygons_union.union(poly)
                continue

            # 去除上一个多边形的覆盖范围
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)

            anno = {}
            anno["iscrowd"] = label_name.endswith("group")
            anno["category_id"] = _get_name2id_dict()[label.name]

            if isinstance(poly_wo_overlaps, Polygon):
                poly_list = [poly_wo_overlaps]
            elif isinstance(poly_wo_overlaps, MultiPolygon):
                poly_list = poly_wo_overlaps.geoms
            else:
                raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))

            poly_coord = []
            for poly_el in poly_list:
                poly_coord.append(list(chain(*poly_el.exterior.coords)))
            anno["segmentation"] = poly_coord
            (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            BASE_CLASSES, NOVEL_CLASSES, ALL_CLASSES, CLASS_NAMES = get_openset_settings_cityscapes(openset_setting)

            if is_train:
                if label.name in BASE_CLASSES:
                    annos.append(anno)
            else:
                if label.name in BASE_CLASSES:
                    annos.append(anno)
                else:
                    anno['category_id']=_get_name2id_dict()['unknown']
                    annos.append(anno)

    else:
        # See also the official annotation parsing scripts at
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
        with PathManager.open(instance_id_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        flattened_ids = np.unique(inst_image[inst_image >= 24])

        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        for instance_id in flattened_ids:
            # For non-crowd annotations, instance_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            anno = {}
            anno["iscrowd"] = instance_id < 1000
            anno["category_id"] = label.id

            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            if to_polygons:
                # This conversion comes from D4809743 and D5171122,
                # when Mask-RCNN was first developed.
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                    -2
                ]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                # opencv's can produce invalid polygons
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
            annos.append(anno)
    ret["annotations"] = annos
    return ret

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



# def main() -> None:
#     global logger, labels
#     """
#     Test the cityscapes dataset loader.
#
#     Usage:
#         python -m detectron2.data.datasets.cityscapes \
#             cityscapes/leftImg8bit/train cityscapes/gtFine/train
#     """
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_dir")
#     parser.add_argument("gt_dir")
#     parser.add_argument("--type", choices=["instance", "semantic"], default="instance")
#     args = parser.parse_args()
#     from cityscapesscripts.helpers.labels import labels
#     from detectron2.data.catalog import Metadata
#     from detectron2.utils.visualizer import Visualizer
#
#     logger = setup_logger(name=__name__)
#
#     dirname = "cityscapes-data-vis"
#     os.makedirs(dirname, exist_ok=True)
#
#     if args.type == "instance":
#         dicts = load_cityscapes_instances(
#             args.image_dir, args.gt_dir, from_json=True, to_polygons=True
#         )
#         logger.info("Done loading {} samples.".format(len(dicts)))
#
#         thing_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]
#         meta = Metadata().set(thing_classes=thing_classes)
#
#     else:
#         dicts = load_cityscapes_semantic(args.image_dir, args.gt_dir)
#         logger.info("Done loading {} samples.".format(len(dicts)))
#
#         stuff_classes = [k.name for k in labels if k.trainId != 255]
#         stuff_colors = [k.color for k in labels if k.trainId != 255]
#         meta = Metadata().set(stuff_classes=stuff_classes, stuff_colors=stuff_colors)
#
#     for d in dicts:
#         img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
#         visualizer = Visualizer(img, metadata=meta)
#         vis = visualizer.draw_dataset_dict(d)
#         # cv2.imshow("a", vis.get_image()[:, :, ::-1])
#         # cv2.waitKey()
#         fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
#         vis.save(fpath)
#
#
# if __name__ == "__main__":
#     main()  # pragma: no cover
