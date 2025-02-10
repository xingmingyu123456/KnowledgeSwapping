# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import logging
import multiprocessing as mp
from collections import deque

import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from swapping_12_07_test import Trainer_forget
class CustomPredictor(DefaultPredictor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.eval()

        state_dict = self.model.state_dict()

        dataset_name = cfg.DATASETS.TEST[0]
        if dataset_name == "remain_dataset_val":
            self.pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            self.pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)
        elif dataset_name == "learn_dataset_val":
            # voc 数据集 感觉没区别
            # self.pixel_mean = torch.tensor([117.79723406, 113.51956767, 98.64946676], dtype=torch.float32).view(3, 1, 1)
            # self.pixel_std = torch.tensor([58.52638834,58.14435355,59.14101012], dtype=torch.float32).view(3, 1, 1)
            # oxofrd pet数据集
            self.pixel_mean = torch.tensor([119.73433564, 109.83269262, 94.84763627], dtype=torch.float32).view(3, 1, 1)
            self.pixel_std = torch.tensor([57.6707407, 57.19914482, 58.46454528], dtype=torch.float32).view(3, 1, 1)

        elif dataset_name == "forget_dataset_val":
            self.pixel_mean = torch.tensor([123.675, 116.280, 103.530], dtype=torch.float32).view(3, 1, 1)
            self.pixel_std = torch.tensor([58.395, 57.120, 57.375], dtype=torch.float32).view(3, 1, 1)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        # self.pixel_mean = self.pixel_mean.to(self.cfg.MODEL.DEVICE)
        # self.pixel_std = self.pixel_std.to(self.cfg.MODEL.DEVICE)

    def custom_load(self, path, resume=False):
        """
        自定义加载函数
        """
        logger = logging.getLogger("detectron2.trainer")
        logger.info(f"Loading checkpoint from: {path}")

        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        # 加载模型权重
        # self.model.load_state_dict(checkpoint["model"],strict=False)
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            missing_keys, unexpected_keys = self.model.module.load_state_dict(checkpoint["model"], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model"], strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if resume:
            # 加载优化器状态
            if "optimizer" in checkpoint and self.optimizer:
                logger.info("Loading optimizer from checkpoint")
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            # 加载调度器状态
            if "scheduler" in checkpoint and self.scheduler:
                logger.info("Loading scheduler from checkpoint")
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            # 恢复迭代次数
            if "iteration" in checkpoint:
                self.iter = checkpoint["iteration"]

        return checkpoint

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            image = (image - self.pixel_mean) / self.pixel_std

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions
class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            trainer = Trainer_forget(cfg)
            trainer.custom_load(cfg.MODEL.WEIGHTS, resume=False)  # 直接加载指定权重
            trainer.model.eval()
            self.predictor = CustomPredictor(cfg)
            self.predictor.model = trainer.model
            # self.predictor.custom_load(cfg.MODEL.WEIGHTS,resume=False)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode,font_size_scale=1.5)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
