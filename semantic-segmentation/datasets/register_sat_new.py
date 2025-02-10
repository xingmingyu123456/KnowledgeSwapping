from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image
import os
def load_semantic_segmentation_dataset(image_dir, annotation_dir):
    dataset_dicts = []
    for idx, image_file in enumerate(os.listdir(image_dir)):
        record = {}

        # 图片路径和标注路径
        file_name = os.path.join(image_dir, image_file)
        if image_file.endswith('.jpg'):
            sem_seg_file_name = os.path.join(annotation_dir, image_file.replace('_sat.jpg', '_mask.png'))
        else:
            sem_seg_file_name = os.path.join(annotation_dir, image_file)
        path = os.path.join(image_dir, image_file)
        img=Image.open(path)
        # 获取图片尺寸
        width, height = img.size # 假设图片大小固定或提前已知
        record["file_name"] = file_name
        record["sem_seg_file_name"] = sem_seg_file_name
        record["height"] = height
        record["width"] = width
        record["image_id"] = idx  # 唯一 ID


        dataset_dicts.append(record)
    return dataset_dicts

def register_sat_new(image_dir,annotation_dir,image_dir_val,annotation_dir_val):
    # thing_classes=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    #  'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    #  'x', 'y', "urban_land","agriculture_land","rangeland",
    #                "forest_land","water","barren_land"]
    # thing_classes = [
    #     "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    #     "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
    #     "car", "truck", "bus", "train", "motorcycle", "bicycle", "a", "b", "c",
    #     "d", "e", "f", "urban_land", "agriculture_land", "rangeland",
    #     "forest_land", "water", "barren_land"]
    # thing_classes = [
    #     "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    #     "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
    #     "car", "truck", "bus", "train", "motorcycle", "bicycle", "sheep", "elephant", "bear",
    #     "zebra", "giraffe", "a", "b", "c", "d",
    #     "e", "f", "g"]
    # thing_classes=["urban_land","agriculture_land","rangeland",
    #                "forest_land","water","barren_land"]
    # thing_dataset_id_to_contiguous_id = {25:0, 26:1,27:2,28:3,29:4,30:5}  # 原始ID映射到Detectron2连续ID
    # thing_classes = [
    #     "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    #     "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
    #     "car", "truck", "bus", "train", "motorcycle", "bicycle", "a", "b", "c",
    #     "d", "e","bird","cat","cow","dog","horse","sheep","g"]
    thing_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window ', 'grass',
                     'cabinet',
                     'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                     'curtain', 'chair',
                     'car', 'water', 'picture', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
                     'armchair', 'seat',
                     'fence', 'desk', 'rock', 'closet', 'lamp', 'tub', 'rail', 'cushion',
                     'pedestal', 'box', 'pillar', 'sign',
                     'dresser',
                     'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
                     'grandstand', 'path',
                     'stairs', 'runway', 'case',
                     'pool table', 'pillow',
                     'screen', 'staircase', 'river', 'bridge', 'bookcase', 'blind',
                     'coffee table',
                     'toilet', 'flower', 'book', 'hill', 'bench',
                     'countertop',
                     'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
                     'arcade machine',
                     'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier',
                     'sunshade', 'street lamp', 'booth', 'tv', 'plane', 'dirt track', 'clothes',
                     'pole',
                     'land', 'bannister',
                     'escalator', 'pouf', 'bottle',
                     'sideboard', 'poster', 'stage', 'van',
                     'ship', 'fountain',
                     'conveyer belt', 'canopy',
                     'washer', 'toy', 'pool', 'stool', 'barrel, cask',
                     'basket', 'falls', 'tent', 'bag', 'motorbike', 'cradle', 'oven', 'ball',
                     'food',
                     'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
                     'dishwasher',
                     'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                     'tray',
                     'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower',
                     'radiator',
                     'glass', 'clock', 'flag',"urban_land","agriculture_land","rangeland","forest_land","water","barren_land","a","b","c","d"]

    thing_colors = [(102, 179, 92), (14, 106, 71), (188, 20, 102), (121, 210, 214), (74, 202, 87), (116, 99, 103), (151, 130, 149), (52, 1, 87), (235, 157, 37), (129, 191, 187), (20, 160, 203), (57, 21, 252), (235, 88, 48), (218, 58, 254), (169, 255, 219), (187, 207, 14), (189, 189, 174), (189, 50, 107), (54, 243, 63), (248, 130, 228), (50, 134, 20), (72, 166, 17), (131, 88, 59), (13, 241, 249), (8, 89, 52), (129, 83, 91), (110, 187, 198), (171, 252, 7), (174, 34, 205), (80, 163, 49), (103, 131, 1), (253, 133, 53), (105, 3, 53), (220, 190, 145), (217, 43, 161), (201, 189, 227), (13, 94, 47), (14, 199, 205), (214, 251, 248), (189, 39, 212), (207, 236, 81), (110, 52, 23), (153, 216, 251), (187, 123, 236), (40, 156, 14), (44, 64, 88), (70, 8, 87), (128, 235, 135), (215, 62, 138), (242, 80, 135), (162, 162, 32), (122, 4, 233), (230, 249, 40), (27, 134, 200), (71, 11, 161), (32, 47, 246), (150, 61, 215), (36, 98, 171), (103, 213, 218), (34, 192, 226), (100, 174, 205), (130, 0, 4), (217, 246, 254), (141, 102, 26), (136, 206, 14), (89, 41, 123), (204, 178, 62), (95, 230, 240), (51, 252, 95), (131, 221, 228), (150, 230, 236), (142, 170, 28), (35, 12, 159), (70, 186, 242), (85, 27, 65), (169, 44, 61), (184, 244, 133), (27, 27, 107), (43, 83, 29), (189, 74, 127), (249, 246, 91), (216, 230, 189), (224, 128, 120), (26, 189, 120), (115, 204, 232), (2, 102, 197), (199, 154, 136), (61, 164, 224), (50, 233, 171), (151, 206, 58), (117, 159, 95), (215, 232, 179), (112, 61, 240), (185, 51, 11), (253, 38, 129), (130, 112, 100), (112, 183, 80), (186, 112, 1), (129, 219, 53), (86, 228, 223), (224, 128, 146), (125, 129, 52), (171, 217, 159), (197, 159, 246), (67, 182, 202), (183, 122, 144), (254, 37, 23), (68, 115, 97), (197, 213, 138), (254, 239, 143), (96, 200, 123), (186, 69, 207), (92, 2, 147), (251, 186, 163), (146, 89, 194), (254, 146, 147), (95, 198, 51), (232, 160, 167), (127, 38, 81), (103, 128, 10), (219, 184, 216), (177, 150, 158), (221, 41, 98), (6, 251, 143), (89, 111, 248), (243, 59, 112), (1, 128, 47), (253, 139, 196), (36, 159, 250), (246, 8, 232), (98, 146, 47), (207, 130, 147), (151, 53, 119), (160, 151, 115), (74, 112, 199), (163, 165, 103), (83, 253, 226), (111, 253, 216), (98, 152, 92), (145, 127, 109), (81, 193, 53), (162, 207, 188), (168, 227, 160), (67, 32, 141), (20, 47, 147), (247, 127, 135), (134, 194, 144), (127, 32, 175), (203, 186, 114), (213, 118, 21), (237, 157, 37), (229, 108, 50), (181, 7, 26), (26, 225, 20), (29, 96, 27), (110, 191, 224), (196, 251, 60), (47, 146, 3), (34, 191, 48), (255, 16, 171)]

    DatasetCatalog.register("learn_dataset_train", lambda: load_semantic_segmentation_dataset(image_dir, annotation_dir))
    MetadataCatalog.get("learn_dataset_train").set(
        thing_classes=thing_classes,
        thing_colors = thing_colors,
        stuff_classes=thing_classes,
        # thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        evaluator_type="sem_seg",  # 指定评估类型为语义分割
        image_root=image_dir,
        sem_seg_root=annotation_dir,
        ignore_label=255,
    )

    DatasetCatalog.register("learn_dataset_val", lambda: load_semantic_segmentation_dataset(image_dir_val, annotation_dir_val))
    MetadataCatalog.get("learn_dataset_val").set(
        thing_classes=thing_classes,
        thing_colors = thing_colors,
        stuff_classes=thing_classes,
        # thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        evaluator_type="sem_seg",  # 指定评估类型为语义分割
        image_root=image_dir_val,
        sem_seg_root=annotation_dir_val,
        ignore_label=255,
    )