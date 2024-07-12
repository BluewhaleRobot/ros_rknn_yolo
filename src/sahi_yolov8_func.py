print('load sahi_yolov8_func.py')
#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
import rospy
from vision_msgs.msg import Detection2D,Detection2DArray,ObjectHypothesisWithPose,YoloResult
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from typing import Any, Dict, List, Optional, Tuple

# category_mapping = {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'motorcycle', '4': 'airplane', '5': 'bus',
#                     '6': 'train', '7': 'truck', '8': 'boat', '9': 'traffic light', '10': 'fire hydrant',
#                     '11': 'stop sign', '12': 'parking meter', '13': 'bench', '14': 'bird', '15': 'cat', '16': 'dog',
#                     '17': 'horse', '18': 'sheep', '19': 'cow', '20': 'elephant', '21': 'bear', '22': 'zebra',
#                     '23': 'giraffe', '24': 'backpack', '25': 'umbrella', '26': 'handbag', '27': 'tie',
#                     '28': 'suitcase', '29': 'frisbee', '30': 'skis', '31': 'snowboard', '32': 'sports ball',
#                     '33': 'kite', '34': 'baseball bat', '35': 'baseball glove', '36': 'skateboard',
#                     '37': 'surfboard', '38': 'tennis racket', '39': 'bottle', '40': 'wine glass', '41': 'cup',
#                     '42': 'fork', '43': 'knife', '44': 'spoon', '45': 'bowl', '46': 'banana', '47': 'apple',
#                     '48': 'sandwich', '49': 'orange', '50': 'broccoli', '51': 'carrot', '52': 'hot dog',
#                     '53': 'pizza', '54': 'donut', '55': 'cake', '56': 'chair', '57': 'couch', '58': 'potted plant',
#                     '59': 'bed', '60': 'dining table', '61': 'toilet', '62': 'tv', '63': 'laptop', '64': 'mouse',
#                     '65': 'remote', '66': 'keyboard', '67': 'cell phone', '68': 'microwave', '69': 'oven',
#                     '70': 'toaster', '71': 'sink', '72': 'refrigerator', '73': 'book', '74': 'clock', '75': 'vase',
#                     '76': 'scissors', '77': 'teddy bear', '78': 'hair drier', '79': 'toothbrush'}


# OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640

# CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
#            "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
#            "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
#            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
#            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
#            "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
#            "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

OBJ_THRESH = rospy.get_param('~obj_thresh', 0.25)
NMS_THRESH = rospy.get_param('~nms_thresh', 0.45)
IMG_SIZE = rospy.get_param('~model_img_size', 640)

CLASSES = rospy.get_param('~classes', ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "))

CATEGORY_MAPPING  = rospy.get_param('~category_mapping', {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'motorcycle', '4': 'airplane', '5': 'bus',
                    '6': 'train', '7': 'truck', '8': 'boat', '9': 'traffic light', '10': 'fire hydrant',
                    '11': 'stop sign', '12': 'parking meter', '13': 'bench', '14': 'bird', '15': 'cat', '16': 'dog',
                    '17': 'horse', '18': 'sheep', '19': 'cow', '20': 'elephant', '21': 'bear', '22': 'zebra',
                    '23': 'giraffe', '24': 'backpack', '25': 'umbrella', '26': 'handbag', '27': 'tie',
                    '28': 'suitcase', '29': 'frisbee', '30': 'skis', '31': 'snowboard', '32': 'sports ball',
                    '33': 'kite', '34': 'baseball bat', '35': 'baseball glove', '36': 'skateboard',
                    '37': 'surfboard', '38': 'tennis racket', '39': 'bottle', '40': 'wine glass', '41': 'cup',
                    '42': 'fork', '43': 'knife', '44': 'spoon', '45': 'bowl', '46': 'banana', '47': 'apple',
                    '48': 'sandwich', '49': 'orange', '50': 'broccoli', '51': 'carrot', '52': 'hot dog',
                    '53': 'pizza', '54': 'donut', '55': 'cake', '56': 'chair', '57': 'couch', '58': 'potted plant',
                    '59': 'bed', '60': 'dining table', '61': 'toilet', '62': 'tv', '63': 'laptop', '64': 'mouse',
                    '65': 'remote', '66': 'keyboard', '67': 'cell phone', '68': 'microwave', '69': 'oven',
                    '70': 'toaster', '71': 'sink', '72': 'refrigerator', '73': 'book', '74': 'clock', '75': 'vase',
                    '76': 'scissors', '77': 'teddy bear', '78': 'hair drier', '79': 'toothbrush'})

np.random.seed(0)
color_palette = np.random.uniform(100, 255, size=(len(CLASSES), 3))

HIDE_LABEL = rospy.get_param('~hide_label', False)  # 是否隐藏标签
SLICE_HEIGHT = rospy.get_param('~slice_height', 480)  # 切片高度
SLICE_WIDTH = rospy.get_param('~slice_width', 480)  # 切片宽度
OVERLAP_HEIGHT_RATIO = rospy.get_param('~overlap_height_ratio', 0.25)  # 高度重叠比率
OVERLAP_WIDTH_RATIO = rospy.get_param('~overlap_width_ratio', 0.25)  # 宽度重叠比率

POSTPROCESS_TYPE = rospy.get_param('~postprocess_type', 'GREEDYNMM')
POSTPROCESS_MATCH_METRIC = rospy.get_param('~postprocess_match_metric', 'IOS')
POSTPROCESS_MATCH_THRESHOLD = rospy.get_param('~postprocess_match_threshold', 0.5)
POSTPROCESS_CLASS_AGNOSTIC = rospy.get_param('~postprocess_class_agnostic', False)
            
def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    # x = np.array(position)
    n,c,h,w = position.shape
    p_num = 4
    mc = c//p_num
    y = position.reshape(n,p_num,mc,h,w)
    
    # Vectorized softmax
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # subtract max for numerical stability
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    
    acc_metrix = np.arange(mc).reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y
    

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE//grid_h, IMG_SIZE//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, object_prediction_list):
    rect_th = max(round(sum(image.shape) / 2 * 0.003), 2)
    for object_prediction in object_prediction_list:
        box = object_prediction.bbox
        score = object_prediction.score.value
        cl = object_prediction.category.id

        top, left, right, bottom = box.minx, box.miny, box.maxx, box.maxy
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        color = color_palette[cl]
        
        cv2.rectangle(image, (top, left), (int(right), int(bottom)), color, rect_th)
        if not HIDE_LABEL:
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, max(rect_th-1 ,1))

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    #return im
    return im, ratio, (left, top)

class DetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = False,
        image_size: int = None,
    ):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)

    def check_dependencies(self) -> None:
        """
        This function can be implemented to ensure model dependencies are installed.
        """
        pass

    def set_model(self, model: Any, **kwargs):
        """
        This function should be implemented to instantiate a DetectionModel out of an already loaded model
        Args:
            model: Any
                Loaded model
        """
        raise NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = None

    def perform_inference(self, image: np.ndarray):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
        """
        raise NotImplementedError()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        This function should be implemented in a way that self._original_predictions should
        be converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list. self.mask_threshold can also be utilized.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        raise NotImplementedError()

    def _apply_category_remapping(self):
        """
        Applies category remapping based on mapping given in self.category_remapping
        """
        # confirm self.category_remapping is not None
        if self.category_remapping is None:
            raise ValueError("self.category_remapping cannot be None")
        # remap categories
        for object_prediction_list in self._object_prediction_list_per_image:
            for object_prediction in object_prediction_list:
                old_category_id_str = str(object_prediction.category.id)
                new_category_id_int = self.category_remapping[old_category_id_str]
                object_prediction.category.id = new_category_id_int

    def convert_original_predictions(
        self,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
        )
        if self.category_remapping:
            self._apply_category_remapping()

    @property
    def object_prediction_list(self):
        return self._object_prediction_list_per_image[0]

    @property
    def object_prediction_list_per_image(self):
        return self._object_prediction_list_per_image

    @property
    def original_predictions(self):
        return self._original_predictions

class Yolov8RknnDetectionModel(DetectionModel):
    def __init__(self, *args, iou_threshold: float = 0.7, **kwargs):
        """
        Args:
            iou_threshold: float
                IOU threshold for non-max supression, defaults to 0.7.
        """
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold


    def set_model(self, model: Any) -> None:
        """
        Sets the underlying ONNX model.

        Args:
            model: Any
                A ONNX model
        """
        self.model = model
        # set category_mapping
        if not self.category_mapping:
            raise TypeError("Category mapping values are required")

    def _preprocess_image(self, image: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
        """Prepapre image for inference by resizing, normalizing and changing dimensions.

        Args:
            image: np.ndarray
                Input image with color channel order RGB.
        """
        input_image = cv2.resize(image, input_shape)
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def _post_process(
        self, outputs: np.ndarray, input_shape: Tuple[int, int], image_shape: Tuple[int, int]
    ):
        # Format the results
        prediction_result = []
        
        boxes, class_ids, scores = yolov8_post_process(outputs)
        image_h, image_w = image_shape
        input_w, input_h = input_shape
        # Scale boxes to original dimensions
        #判断boxes这个np数组是否为空
        if boxes is None:
            prediction_result = [prediction_result]
            return prediction_result
        boxes = boxes * np.array([image_w / input_w, image_h / input_h, image_w / input_w, image_h / input_h])
        # boxes 取整数
        boxes = np.round(boxes).astype(np.int32)
        
        for bbox, score, label in zip(boxes, scores, class_ids):
            bbox = bbox.tolist()
            cls_id = int(label)
            prediction_result.append([bbox[0], bbox[1], bbox[2], bbox[3], float(score), cls_id])
        # prediction_result = [torch.tensor(prediction_result)]
        prediction_result = [prediction_result]
        return prediction_result

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        input_shape = (IMG_SIZE,IMG_SIZE)  # w, h
        image_shape = image.shape[:2]  # h, w
        # Prepare image
        input_image = self._preprocess_image(image, input_shape)
        # Inference
        outputs = self.model.inference(inputs=[input_image],data_format=['nhwc'])
        # Post-process
        prediction_results = self._post_process(outputs, input_shape, image_shape)
        self._original_predictions = prediction_results

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []
            # process predictions
            # for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
            for prediction in image_predictions_in_xyxy_format:
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]
                # category_name = classes[category_id]
                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])
                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])
                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    print(f"ignoring invalid prediction with bbox: {bbox}")
                    continue
                try:
                    object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                except Exception as e:
                    print(f"Caught an exception: {e}")
                
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
        self._object_prediction_list_per_image = object_prediction_list_per_image
        
def rknn_Func(rknn_lite,  bridge, IMG, image_header, Crop_object_flag = False, Draw_flag=False):
    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    # 初始化YOLOv8模型
    yolov8_rknn_detection_model = Yolov8RknnDetectionModel(
        model=rknn_lite,  # 模型路径
        confidence_threshold=OBJ_THRESH,  # 置信度阈值
        iou_threshold=NMS_THRESH,  # 交并比阈值
        category_mapping=CATEGORY_MAPPING,  # 类别映射
        load_at_init=True,  # 初始化时加载模型
        image_size=IMG_SIZE,  # 图像尺寸
    )
    
    sahi_result = get_sliced_prediction(
            IMG2,
            yolov8_rknn_detection_model,
            slice_height = SLICE_HEIGHT,  # 切片高度
            slice_width = SLICE_WIDTH,  # 切片宽度
            overlap_height_ratio = OVERLAP_HEIGHT_RATIO,  # 高度重叠比率
            overlap_width_ratio = OVERLAP_WIDTH_RATIO,  # 宽度重叠比率
            postprocess_type = POSTPROCESS_TYPE,
            postprocess_match_metric = POSTPROCESS_MATCH_METRIC,
            postprocess_match_threshold = POSTPROCESS_MATCH_THRESHOLD,
            postprocess_class_agnostic = POSTPROCESS_CLASS_AGNOSTIC,
        )
    #print type of sahi_result.object_prediction_list
    # print(type(sahi_result.object_prediction_list[0].bbox))
    
    yolo_result_msg = YoloResult()
    yolo_result_msg.header = image_header

    if sahi_result.object_prediction_list:
        for object_prediction in sahi_result.object_prediction_list:
            detection = Detection2D()
            detection.bbox.center.x = float((object_prediction.bbox.minx + object_prediction.bbox.maxx) / 2.0)
            detection.bbox.center.y = float((object_prediction.bbox.miny + object_prediction.bbox.maxy) / 2.0)
            detection.bbox.size_x = float(object_prediction.bbox.maxx - object_prediction.bbox.minx)
            detection.bbox.size_y = float(object_prediction.bbox.maxy - object_prediction.bbox.miny)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(object_prediction.category.id)
            hypothesis.score = float(object_prediction.score.value)
            detection.results.append(hypothesis)
            detection.object_name = object_prediction.category.name
            if Crop_object_flag:
                #根据检测框裁剪目标物体图像
                crop_img = IMG[object_prediction.bbox.miny:object_prediction.bbox.maxy, object_prediction.bbox.minx:object_prediction.bbox.maxx]
                #用cv_bridge将裁剪的目标物体图像转换为ros的sensor_msgs/Image消息格式
                detection.source_img = bridge.cv2_to_imgmsg(crop_img, encoding="bgr8")
            yolo_result_msg.detections.append(detection)
    
    if sahi_result.object_prediction_list and Draw_flag:
        draw(IMG, sahi_result.object_prediction_list)
    return yolo_result_msg, IMG
