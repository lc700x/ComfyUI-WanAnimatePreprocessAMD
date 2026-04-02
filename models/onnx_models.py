# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import cv2
import numpy as np
import torch
import onnxruntime
import threading

from ..pose_utils.pose2d_utils import box_convert_simple, keypoints_from_heatmaps


class SimpleOnnxInference(object):
    """
    Thread-safe ONNX inference wrapper.

    Uses:
    - per-thread sessions
    - session lifecycle lock
    """

    def __init__(self, checkpoint, device='CPUExecutionProvider', **kwargs):
        self.checkpoint = checkpoint
        self.init_kwargs = kwargs

        if device != "CPUExecutionProvider":
            provider = [device, "CPUExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        self.provider = provider

        self._session_lock = threading.RLock()
        self._thread_local = threading.local()

        self._create_base_session()

    def _create_base_session(self):
        with self._session_lock:
            self.session = onnxruntime.InferenceSession(
                self.checkpoint,
                providers=self.provider,
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_resolution = np.array(self.session.get_inputs()[0].shape[2:])

    def _get_session(self):
        if not hasattr(self._thread_local, "session"):
            with self._session_lock:
                self._thread_local.session = onnxruntime.InferenceSession(
                    self.checkpoint,
                    providers=self.provider,
                )
        return self._thread_local.session

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_output_names(self):
        session = self._get_session()
        return [node.name for node in session.get_outputs()]

    def cleanup(self):
        with self._session_lock:
            if hasattr(self, "session") and self.session is not None:
                del self.session
                self.session = None

            if hasattr(self._thread_local, "session"):
                del self._thread_local.session

    def reinit(self, provider=None):
        with self._session_lock:
            if provider is not None:
                self.provider = provider

            self.cleanup()
            self._create_base_session()


class Yolo(SimpleOnnxInference):
    def __init__(
        self,
        checkpoint,
        device='CPUExecutionProvider',
        threshold_conf=0.05,
        threshold_multi_persons=0.1,
        input_resolution=(640, 640),
        threshold_iou=0.5,
        threshold_bbox_shape_ratio=0.4,
        cat_id=[1],
        select_type='max',
        strict=True,
        sorted_func=None,
        **kwargs
    ):
        super().__init__(checkpoint, device=device, **kwargs)

        self.input_width = 640
        self.input_height = 640

        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func

    def postprocess(self, output, shape_raw, cat_id=[1]):
        outputs = np.squeeze(output)

        if len(outputs.shape) == 1:
            outputs = outputs[None]

        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)

        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]

            threshold_conf_masks = max_scores >= self.threshold_conf
            classid_masks = classid[threshold_conf_masks] != 3.14159

            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[threshold_conf_masks][classid_masks]

            boxes = outputs[:, :4][threshold_conf_masks][classid_masks]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)

        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            threshold_conf_masks = max_scores >= self.threshold_conf

            classid = np.argmax(classes_scores[threshold_conf_masks], -1)
            classid_masks = classid != 3.14159

            classes_scores = classes_scores[threshold_conf_masks][classid_masks]
            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[classid_masks]

            xywh = outputs[:, :4][threshold_conf_masks][classid_masks]

            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]

            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()

        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.threshold_conf,
            self.threshold_iou,
        )

        if indices is None or len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)

        indices = np.array(indices).reshape(-1).tolist()

        results = []
        for i in indices:
            box = box_convert_simple(boxes[i], 'xywh2xyxy')
            score = scores[i]
            class_id = class_ids[i]
            results.append(box + [score] + [class_id])

        return np.array(results, dtype=np.float32)

    def process_results(self, results, shape_raw, cat_id=[1], single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results

        person_results = []
        person_count = 0

        if det_results is None or len(det_results) == 0:
            return None

        max_idx = -1
        max_bbox_size = shape_raw[0] * shape_raw[1] * -10
        max_bbox_shape = -1

        bboxes = []
        idx_list = []

        for i in range(det_results.shape[0]):
            bbox = det_results[i]
            if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                idx_list.append(i)
                bbox_shape = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1]))
                if bbox_shape > max_bbox_shape:
                    max_bbox_shape = bbox_shape

        if len(idx_list) == 0:
            return None

        det_results = det_results[idx_list]

        for i in range(det_results.shape[0]):
            bbox = det_results[i]
            bboxes.append(bbox)

            if self.select_type == 'max':
                bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
            elif self.select_type == 'center':
                bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1
            else:
                bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))

            bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))

            if bbox_size > max_bbox_size:
                if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                    continue
                max_bbox_size = bbox_size
                max_bbox_shape = bbox_shape
                max_idx = i

        if self.sorted_func is not None and len(bboxes) > 0:
            max_idx = self.sorted_func(bboxes, shape_raw)
            bbox = bboxes[max_idx]
            if self.select_type == 'max':
                max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
            elif self.select_type == 'center':
                max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1

        if max_idx != -1:
            person_count = 1

        if max_idx != -1:
            person = {}
            person['bbox'] = det_results[max_idx, :5]
            person['track_id'] = int(0)
            person_results.append(person)

        for i in range(det_results.shape[0]):
            bbox = det_results[i]
            if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1] / 2)) * -1
                else:
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))

                if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                    person_count += 1
                    if not single_person:
                        person = {}
                        person['bbox'] = det_results[i, :5]
                        person['track_id'] = int(person_count - 1)
                        person_results.append(person)

        return person_results if len(person_results) > 0 else None

    def postprocess_threading(self, outputs, shape_raw, person_results, i, single_person=True, **kwargs):
        result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
        result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, single_person=single_person)
        if result is not None and len(result) != 0:
            person_results[i] = result

    def forward(self, img, shape_raw, **kwargs):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()

        session = self._get_session()

        outputs = session.run(
            None,
            {session.get_inputs()[0].name: img}
        )[0]

        person_results = [
            [{
                'bbox': np.array([
                    0.,
                    0.,
                    1. * shape_raw[i][1],
                    1. * shape_raw[i][0],
                    -1
                ]),
                'track_id': -1
            }]
            for i in range(len(outputs))
        ]

        for i in range(len(outputs)):
            self.postprocess_threading(
                outputs,
                shape_raw,
                person_results,
                i,
                **kwargs
            )

        return person_results


class ViTPose(SimpleOnnxInference):
    def __init__(self, checkpoint, device='CPUExecutionProvider', **kwargs):
        super().__init__(checkpoint, device=device)

    def forward(self, img, center, scale, **kwargs):
        session = self._get_session()

        heatmaps = session.run(
            [],
            {session.get_inputs()[0].name: img}
        )[0]

        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=center,
            scale=scale * 200,
            unbiased=True,
            use_udp=False
        )

        return np.concatenate([points, prob], axis=2)