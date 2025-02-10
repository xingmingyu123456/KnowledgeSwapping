from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image
import os
import cv2
from PIL import Image
def load_semantic_segmentation_dataset(image_dir, annotation_dir):
    dataset_dicts = []
    for sub in os.listdir(image_dir):
        image_subdir = os.path.join(image_dir, sub)
        annotation_subdir = os.path.join(annotation_dir, sub)
        for image_filename in os.listdir(image_subdir):
            record = {}

            # 构建图像文件路径和对应的分割标签路径
            image_path = os.path.join(image_subdir, image_filename)
            # bochum_000000_000313_gtFine_labelTrainIds.png  bochum_000000_000313_leftImg8bit.png
            label_path = os.path.join(annotation_subdir, image_filename.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png"))

            # 读取图像大小
            # height, width = cv2.imread(image_path).shape[:2]
            img=Image.open(image_path)
            width,height = img.size

            record["file_name"] = image_path
            record["sem_seg_file_name"] = label_path  # 语义分割标签路径
            record["height"] = height
            record["width"] = width
            dataset_dicts.append(record)

    return dataset_dicts

def register_forget(image_dir,annotation_dir,image_dir_val,annotation_dir_val):
    # thing_classes = [
    # "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    # "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
    # "car", "truck", "bus", "train", "motorcycle", "bicycle","a","b","c",
    #     "d","e","f","urban_land","agriculture_land","rangeland",
    #                "forest_land","water","barren_land"]
    # thing_classes = [
    #     "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    #     "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
    #     "car", "truck", "bus", "train", "motorcycle", "bicycle", "sheep", "elephant", "bear",
    #     "zebra", "giraffe", "a", "b", "c", "d",
    #     "e", "f", "g"]
    thing_classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
        "car", "truck", "bus", "train", "motorcycle", "bicycle", "a", "b", "c",
        "d", "e", "bird", "cat", "cow", "dog", "horse", "sheep", "g"]
    DatasetCatalog.register("forget_dataset_train", lambda: load_semantic_segmentation_dataset(image_dir, annotation_dir))
    MetadataCatalog.get("forget_dataset_train").set(
        thing_classes=thing_classes,
        stuff_classes=thing_classes,
        evaluator_type="sem_seg",  # 指定评估类型为语义分割
        image_root=image_dir,
        sem_seg_root=annotation_dir,
        ignore_label=255,
    )
    DatasetCatalog.register("forget_dataset_val", lambda: load_semantic_segmentation_dataset(image_dir_val, annotation_dir_val))
    MetadataCatalog.get("forget_dataset_val").set(
        thing_classes=thing_classes,
        stuff_classes=thing_classes,
        evaluator_type="sem_seg",  # 指定评估类型为语义分割
        image_root=image_dir_val,
        sem_seg_root=annotation_dir_val,
        ignore_label=255,
    )