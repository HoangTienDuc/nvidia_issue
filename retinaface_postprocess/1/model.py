import triton_python_backend_utils as pb_utils
import sys
import json
import numpy as np
from numba import njit, jit
import time
import cv2
from skimage import transform as trans
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)s - %(message)s')

@njit()
def nms(dets, thresh=0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# @jit()
def anchors_plane(height, width, stride, base_anchors):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    A = base_anchors.shape[0]
    all_anchors = np.zeros((height, width, A, 4), dtype=np.float32)
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
    return all_anchors


def clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [n, c, H, W]
    :param pad_shape: [h, w]
    :return: [n, c, h, w]
    """
    H, W = tensor.shape[2:]
    h, w = pad_shape

    if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

    return tensor


def bbox_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))
    boxes = np.array(boxes)
    box_deltas = np.array(box_deltas)
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1] > 4:
        pred_boxes[:, 4:] = box_deltas[:, 4:]

    return pred_boxes


def landmark_pred(boxes, landmark_deltas):
    boxes = np.array(boxes)
    landmark_deltas = np.array(landmark_deltas)
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
    return pred

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = np.array(ws)
    hs = np.array(hs)
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

# @jit()
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    anchor = np.array(anchor)
    ratios = np.array(ratios)
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.around(np.sqrt(size_ratios))
    hs = np.around(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# @jit()
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6), stride=16):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors

# @jit()
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# @jit()
def generate_anchors_fpn(cfg):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # RPN_FEAT_STRIDE = []
    # for k in cfg:
    #     RPN_FEAT_STRIDE.append(int(k))
    # RPN_FEAT_STRIDE = sorted(RPN_FEAT_STRIDE, reverse=True)
    RPN_FEAT_STRIDE = np.array([32, 16, 8])
    anchors = []
    for k in RPN_FEAT_STRIDE:
        v = cfg[str(k)]
        bs = v['BASE_SIZE']
        __ratios = np.array(v['RATIOS'])
        __scales = np.array(v['SCALES'])
        stride = int(k)
        # print('anchors_fpn', bs, __ratios, __scales, file=sys.stderr)
        r = generate_anchors(bs, __ratios, __scales, stride)
        # print('anchors_fpn', r.shape, file=sys.stderr)
        anchors.append(r)

    return anchors

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def __init__(self):
        _ratio = (1.,)
        self.landmark_std = 0.2
        self.masks = True
        self.nms_threshold = 0.4
        self.pupil_distance_thresh = 26

        self._feat_stride_fpn = [32, 16, 8]
        self.anchor_cfg = {
            '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
        }

        self.fpn_keys = []
        self.use_landmarks = True

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
        self.anchor_plane_cache = {}

        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

        src1 = np.array([
            [51.642,50.115],
            [57.617,49.990],
            [35.740,69.007],
            [51.157,89.050],
            [57.025,89.702]], dtype=np.float32)
        #<--left 
        src2 = np.array([
            [45.031,50.118],
            [65.568,50.872],
            [39.677,68.111],
            [45.177,86.190],
            [64.246,86.758]], dtype=np.float32)

        #---frontal
        src3 = np.array([
            [39.730,51.138],
            [72.270,51.138],
            [56.000,68.493],
            [42.463,87.010],
            [69.537,87.010]], dtype=np.float32)

        #-->right
        src4 = np.array([
            [46.845,50.872],
            [67.382,50.118],
            [72.737,68.111],
            [48.167,86.758],
            [67.236,86.190]], dtype=np.float32)

        #-->right profile
        src5 = np.array([
            [54.796,49.990],
            [60.771,50.115],
            [76.673,69.007],
            [55.388,89.702],
            [61.257,89.050]], dtype=np.float32)

        src = np.array([src1,src2,src3,src4,src5])
        self.src_map = {112 : src, 224 : src*2}

        arcface_src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041] ], dtype=np.float32 )

        self.arcface_src = np.expand_dims(arcface_src, axis=0)


    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "det")

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "aligned_images")
        
        output2_config = pb_utils.get_output_config_by_name(
            model_config, "batch_mapping")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output2_config['data_type'])

        
        print('Initialized...')

    def reproject_points(self, dets, scale: float):
        if scale != 1.0:
            dets = dets / scale
        return dets

    # # lmk is prediction; src is template
    # def estimate_norm(self, lmk, image_size = 112, mode='arcface'):
    #     assert lmk.shape==(5,2), print(lmk)
    #     tform = trans.SimilarityTransform()
    #     lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    #     min_M = []
    #     min_index = []
    #     min_error = float('inf') 
    #     if mode=='arcface':
    #         assert image_size==112
    #         src = self.arcface_src
    #     else:
    #         src = self.src_map[image_size]
    #     for i in np.arange(src.shape[0]):
    #         tform.estimate(lmk, src[i])
    #         M = tform.params[0:2,:]
    #         results = np.dot(M, lmk_tran.T)
    #         results = results.T
    #         error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2,axis=1)))
    #     #         print(error)
    #         if error< min_error:
    #             min_error = error
    #             min_M = M
    #             min_index = i
    #     return min_M, min_index

    # def norm_crop(self, warpeds, img, landmarks, image_size=112, mode='arcface'):
    #     warpeds = []
    #     for landmark in landmarks:
    #         print()
    #         # M, pose_index = self.estimate_norm(landmark)
    #         # warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    #         warpeds.append(landmark)
    #     return landmarks

    def filter_face_by_pupil_distance(self, det, landmarks):
        _det = []
        _landmarks = []
        for d, landmark in zip(det, landmarks):
            pupil_distance =  abs(landmark[0][0] - landmark[1][0])
            if pupil_distance >= self.pupil_distance_thresh:
                _det.append(d)
                _landmarks.append(landmark)
        return np.array(_det), np.array(_landmarks)

    def postprocess(self, img, net_out, scale_factor):
        threshold = 0.8
        proposals_list = []
        scores_list = []
        mask_scores_list = []
        landmarks_list = []
        t0 = time.time()
        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if self.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            if self.masks:
                idx = _idx * 4

            A = self._num_anchors['stride%s' % s]

            scores = net_out[idx]
            scores = scores[:, A:, :, :]
            idx += 1
            bbox_deltas = net_out[idx]
            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            K = height * width
            key = (height, width, stride)
            if key in self.anchor_plane_cache:
                anchors = self.anchor_plane_cache[key]
            else:

                anchors_fpn = self._anchors_fpn['stride%s' % s]
                anchors = anchors_plane(height, width, stride, anchors_fpn)
                anchors = anchors.reshape((K * A, 4))
                if len(self.anchor_plane_cache) < 100:
                    self.anchor_plane_cache[key] = anchors

            scores = clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            proposals = bbox_pred(anchors, bbox_deltas)

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals_list.append(proposals)
            scores_list.append(scores)

            if self.masks:
                type_scores = net_out[idx + 2]
                mask_scores = type_scores[:, A*2:, :, :]
                mask_scores = clip_pad(mask_scores,(height, width))
                mask_scores = mask_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                mask_scores = mask_scores[order]
                mask_scores_list.append(mask_scores)

            if self.use_landmarks:
                idx += 1
                landmark_deltas = net_out[idx]
                landmark_deltas = clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
                landmark_deltas *= self.landmark_std
                landmarks = landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 112, 112, 3))
            return np.zeros((0, 6)), landmarks

        scores = np.vstack(scores_list)

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]

        if self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)
        if self.masks:
            mask_scores = np.vstack(mask_scores_list)
            mask_scores = mask_scores[order]
            pre_det = np.hstack((proposals[:, 0:4], scores, mask_scores)).astype(np.float32, copy=False)
        else:
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
        keep = nms(pre_det, thresh=self.nms_threshold)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        if self.use_landmarks:
            landmarks = landmarks[keep]
        det = self.reproject_points(det, scale_factor)
        landmarks = self.reproject_points(landmarks, scale_factor)
        det, landmarks = self.filter_face_by_pupil_distance(det, landmarks)
        warpeds = []
        if len(det) > 0:
            for lmk in landmarks:
                assert lmk.shape==(5,2), print(lmk)
                tform = trans.SimilarityTransform()
                lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
                min_M = []
                min_index = []
                min_error = float('inf') 

                src = self.arcface_src
                for i in np.arange(src.shape[0]):
                    tform.estimate(lmk, src[i])
                    M = tform.params[0:2,:]
                    results = np.dot(M, lmk_tran.T)
                    results = results.T
                    error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2,axis=1)))
                #         print(error)
                    if error< min_error:
                        min_error = error
                        min_M = M
                        min_index = i
                warped = cv2.warpAffine(img, min_M, (112, 112), borderValue = 0.0)
                warpeds.append(warped)
        else:
            det = np.zeros((0, 6))
            warpeds = np.zeros((0, 112, 112, 3))
        return det, np.array(warpeds)
    
    def expand_input_dim(self, in_0, in_1, in_2, in_3, in_4, in_5, \
                    in_6, in_7, in_8, in_9, in_10, in_11):
        in_0 = np.expand_dims(in_0, axis=0)
        in_1 = np.expand_dims(in_1, axis=0)
        in_2 = np.expand_dims(in_2, axis=0)
        in_3 = np.expand_dims(in_3, axis=0)
        in_4 = np.expand_dims(in_4, axis=0)
        in_5 = np.expand_dims(in_5, axis=0)
        in_6 = np.expand_dims(in_6, axis=0)
        in_7 = np.expand_dims(in_7, axis=0)
        in_8 = np.expand_dims(in_8, axis=0)
        in_9 = np.expand_dims(in_9, axis=0)
        in_10 = np.expand_dims(in_10, axis=0)
        in_11 = np.expand_dims(in_11, axis=0)
        return [in_0, in_1, in_2, in_3, in_4, in_5, \
                    in_6, in_7, in_8, in_9, in_10, in_11]

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        output2_dtype = self.output2_dtype

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            # Get INPUT0
            imgs = pb_utils.get_input_tensor_by_name(request, "Origin_images_in_postprocessing").as_numpy()
            scaled_factors = pb_utils.get_input_tensor_by_name(request, "preprocessed_scaled_factors").as_numpy()
            in_0s = pb_utils.get_input_tensor_by_name(request, "face_rpn_cls_prob_reshape_stride32").as_numpy()
            in_1s = pb_utils.get_input_tensor_by_name(request, "face_rpn_bbox_pred_stride32").as_numpy()
            in_2s = pb_utils.get_input_tensor_by_name(request, "face_rpn_landmark_pred_stride32").as_numpy()
            in_3s = pb_utils.get_input_tensor_by_name(request, "face_rpn_type_prob_reshape_stride32").as_numpy()
            in_4s = pb_utils.get_input_tensor_by_name(request, "face_rpn_cls_prob_reshape_stride16").as_numpy()
            in_5s = pb_utils.get_input_tensor_by_name(request, "face_rpn_bbox_pred_stride16").as_numpy()
            in_6s = pb_utils.get_input_tensor_by_name(request, "face_rpn_landmark_pred_stride16").as_numpy()
            in_7s = pb_utils.get_input_tensor_by_name(request, "face_rpn_type_prob_reshape_stride16").as_numpy()
            in_8s = pb_utils.get_input_tensor_by_name(request, "face_rpn_cls_prob_reshape_stride8").as_numpy()
            in_9s = pb_utils.get_input_tensor_by_name(request, "face_rpn_bbox_pred_stride8").as_numpy()
            in_10s = pb_utils.get_input_tensor_by_name(request, "face_rpn_landmark_pred_stride8").as_numpy()
            in_11s = pb_utils.get_input_tensor_by_name(request, "face_rpn_type_prob_reshape_stride8").as_numpy()

            tensor_0s = np.zeros((0, 6))
            tensor_1s = np.zeros((0, 112, 112, 3))
            batch_mapping = []
            # idx = 0
            for img, scaled_factor, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11 in \
                zip(imgs, scaled_factors, in_0s, in_1s, in_2s, in_3s, in_4s, in_5s, in_6s, in_7s, in_8s, in_9s, in_10s, in_11s):

                input = self.expand_input_dim(in_0, in_1, in_2, in_3, in_4, in_5, \
                    in_6, in_7, in_8, in_9, in_10, in_11)
                tensor_0, tensor_1 = self.postprocess(img, input, scaled_factor)
    
                tensor_0s = np.vstack((tensor_0s, tensor_0))
                tensor_1s = np.vstack((tensor_1s, tensor_1))
                batch_mapping.append(len(tensor_0))

            tensor_0s = tensor_0s.astype(output0_dtype)
            tensor_1s = tensor_1s.astype(output1_dtype)
            batch_mapping = np.array(batch_mapping).astype(output2_dtype)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0s = pb_utils.Tensor("det",
                                        tensor_0s)
            out_tensor_1s = pb_utils.Tensor("aligned_images",
                                        tensor_1s)
            out_tensor_2s = pb_utils.Tensor("batch_mapping",
                                        batch_mapping)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0s, out_tensor_1s, out_tensor_2s])
            responses.append(inference_response)
        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

