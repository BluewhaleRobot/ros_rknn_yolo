print('load yolov8_seg_func.py')
#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
import time
import rospy
import cv_bridge
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D,Detection2DArray,ObjectHypothesisWithPose,YoloResult
from sensor_msgs.msg import Image

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
IMG_SIZE = rospy.get_param('~img_size', 640)

CLASSES = rospy.get_param('~classes', ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "))

np.random.seed(0)
color_palette = np.random.uniform(100, 255, size=(len(CLASSES), 3))

HIDE_LABEL = rospy.get_param('~hide_label', False)  # 是否隐藏标签

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part

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

# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     import torch
#     x = torch.tensor(position)
#     n,c,h,w = x.shape
#     p_num = 4
#     mc = c//p_num
#     y = x.reshape(n,p_num,mc,h,w)
#     y = y.softmax(2)
#     acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
#     y = (y*acc_metrix).sum(2)
#     return y.numpy()

# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     n, c, h, w = position.shape
#     p_num = 4
#     mc = c // p_num
#     y = position.reshape(n, p_num, mc, h, w)
#     exp_y = np.exp(y)
#     y = exp_y / np.sum(exp_y, axis=2, keepdims=True)
#     acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
#     y = (y * acc_metrix).sum(2)
#     return y

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

def yolov8_seg_post_process(input_data):
    # input_data[0], input_data[4], and input_data[8] are detection box information
    # input_data[1], input_data[5], and input_data[9] are category score information
    # input_data[2], input_data[6], and input_data[10] are confidence score information
    # input_data[3], input_data[7], and input_data[11] are segmentation information
    # input_data[12] is the proto information
    proto = input_data[-1]
    boxes, scores, classes_conf, seg_parts = [], [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))
        seg_parts.append(input_data[pair_per_branch*i+3])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_parts = [sp_flatten(_v) for _v in seg_parts]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_parts = np.concatenate(seg_parts)

    # filter according to threshold
    boxes, classes, scores, seg_parts = filter_boxes(boxes, scores, classes_conf, seg_parts)

    zipped = zip(boxes, classes, scores, seg_parts)
    sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    result = zip(*sort_zipped)
    
    max_nms = 30000
    n = boxes.shape[0]  # number of boxes
    if not n:
        return None, None, None, None
    elif n > max_nms:  # excess boxes
        boxes, classes, scores, seg_parts = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_parts = [np.array(x) for x in result]
    
    initTime = time.time()
    # nms
    nboxes, nclasses, nscores, nseg_parts = [], [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        n = seg_parts[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
            nseg_parts.append(n[keep])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_parts = np.concatenate(nseg_parts)
    
    initTime2 = time.time()
    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_parts.shape[-1], -1)
    seg_imgs = np.matmul(seg_parts, proto)
    seg_imgs = sigmoid(seg_imgs)
    seg_imgs = seg_imgs.reshape(-1, ph, pw)
    
    # print(seg_imgs.shape)
    width_ratio = pw / 640
    height_ratio = ph / 640
    # print(ph, pw, width_ratio, height_ratio)
    # print(boxes.shape, classes.shape, scores.shape, seg_parts.shape)

    downsampled_bboxes = boxes.copy()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    seg_threadhold = 0.5
    initTime3 = time.time()

    seg_imgs_cropped = _crop_mask(seg_imgs, downsampled_bboxes)
    # print(seg_imgs_cropped.shape)
    seg_imgs_resize = np.array([cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR) for img in seg_imgs_cropped])
    seg_imgs = seg_imgs_resize > seg_threadhold
    # print(seg_imgs.shape)

    # seg_imgs_t = torch.tensor(seg_imgs)
    # downsampled_bboxes_t = torch.tensor(downsampled_bboxes)

    # initTime3 = time.time()
    # seg_imgs_t = _crop_mask2(seg_imgs_t,downsampled_bboxes_t)
    # seg_imgs_t = F.interpolate(seg_imgs_t[None], torch.Size([640, 640]), mode='bilinear', align_corners=False)[0]
    # seg_imgs_t = seg_imgs_t.gt_(seg_threadhold)

    # seg_imgs = seg_imgs_t.numpy()

    procssmaskTime = time.time()
    # print('process mask time1: {:.3f}s'.format(initTime2 - initTime))
    # print('process mask time2: {:.3f}s'.format(initTime3 - initTime2)) 
    # print('process mask time3: {:.3f}s'.format(procssmaskTime - initTime3)) 
    return boxes, classes, scores, seg_imgs

# def _crop_mask(masks, boxes):
#     """
#     "Crop" predicted masks by zeroing out everything not in the predicted bbox.
#     Vectorized by Chong (thanks Chong).

#     Args:
#         - masks should be a size [n, h, w] numpy array of masks
#         - boxes should be a size [n, 4] numpy array of bbox coords in relative point form
#     """

#     n, h, w = masks.shape
#     x1, y1, x2, y2 = np.split(boxes, 4, axis=1)  # x1 shape(n,1,1)
#     x1, y1, x2, y2 = [np.expand_dims(coord, axis=(1,2)) for coord in [x1, y1, x2, y2]]  # expand dimensions to match r and c
#     r = np.arange(w)[None, None, :]  # rows shape(1,w,1)
#     c = np.arange(h)[None, :, None]  # cols shape(h,1,1)
    
#     return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def _crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
    r = np.arange(w, dtype=np.float32)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=np.float32)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        #恢复原图尺寸
        top = (top - padding[0])/ratio[0]
        left = (left - padding[1])/ratio[1]
        right = (right - padding[0])/ratio[0]
        bottom = (bottom - padding[1])/ratio[1]
        top = max(0, top)
        left = max(0, left)
        right = min(image.shape[1], right)
        bottom = min(image.shape[0], bottom)
        
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)

        color = color_palette[cl]
        cv2.rectangle(image, (top, left), (int(right), int(bottom)), color, 2)
        if not HIDE_LABEL:
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        
# def merge_seg(image, seg_img, classes):
#     color = Colors()
#     for i in range(len(seg_img)):
#         seg = seg_img[i]
#         seg = seg.astype(np.uint8)
#         seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
#         seg = seg * color(classes[i])
#         seg = seg.astype(np.uint8)
#         image = cv2.add(image, seg)
#     return image

def merge_seg(image, seg_imgs, classes, padding):
    seg_img = np.sum(seg_imgs, axis=0)
    seg = seg_img.astype(np.uint8)
    seg = seg * 128
    seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)

    if padding[1] == 0:
        if padding[0] != 0:
            seg2 = seg[:, padding[0]:-padding[2]]
        else:
            seg2 = seg
    else:
        if padding[0] == 0:
            seg2 = seg[padding[1]:-padding[3], :]
        else:
            seg2 = seg[padding[1]:-padding[3], padding[0]:-padding[2]]

    seg = cv2.resize(seg2, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    image = cv2.add(image, seg)
    return image

def merge_seg2(image, seg_img):
    seg = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
    image = cv2.add(image, seg)
    return image

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
    return im, ratio, (left, top, right, bottom)

def rknn_Func(rknn_lite,  bridge, IMG, image_header, Crop_object_flag = False, Draw_flag=False):
    #initTime = time.time()
    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    # 等比例缩放
    IMG2, ratio, padding = letterbox(IMG2)
    # 强制放缩
    # IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))
    IMG2 = np.expand_dims(IMG2, 0)
    #prepostTime = time.time()
    # print("预处理时间:\t", prepostTime - initTime, "秒")
    outputs = rknn_lite.inference(inputs=[IMG2],data_format=['nhwc'])
    #inferenceTime = time.time()
    # print("推理时间:\t", inferenceTime - prepostTime, "秒")
    #print("oups1",len(outputs))
    #print("oups2",outputs[0].shape)

    boxes, classes, scores, seg_imgs = yolov8_seg_post_process(outputs)
    #postprocessTime = time.time()
    # print("后处理时间:\t", postprocessTime - inferenceTime, "秒")

    yolo_result_msg = YoloResult()
    yolo_result_msg.header = image_header
    
    if boxes is not None:
        for box, score, cl in zip(boxes, scores, classes):
            top_left_x, top_left_y, right_bottom_x,right_bottom_y = box
            #恢复原图尺寸
            top_left_x = (top_left_x - padding[0])/ratio[0]
            top_left_y = (top_left_y - padding[1])/ratio[1]
            right_bottom_x = (right_bottom_x - padding[0])/ratio[0]
            right_bottom_y = (right_bottom_y - padding[1])/ratio[1]
            
            #限制范围不要超出原图大小
            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            right_bottom_x = min(IMG.shape[1], right_bottom_x)
            right_bottom_y = min(IMG.shape[0], right_bottom_y)
            
            detection = Detection2D()
            detection.bbox.center.x = float((top_left_x + right_bottom_x) / 2.0)
            detection.bbox.center.y = float((top_left_y + right_bottom_y) / 2.0)
            detection.bbox.size_x = float(right_bottom_x - top_left_x)
            detection.bbox.size_y = right_bottom_y - top_left_y
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cl)
            hypothesis.score = float(score)
            detection.results.append(hypothesis)
            detection.object_name = CLASSES[cl]
            if Crop_object_flag:
                #根据检测框裁剪目标物体图像
                crop_img = IMG[int(top_left_y):int(right_bottom_y), int(top_left_x):int(right_bottom_x)]
                #用cv_bridge将裁剪的目标物体图像转换为ros的sensor_msgs/Image消息格式
                detection.source_img = bridge.cv2_to_imgmsg(crop_img, encoding="bgr8")

            yolo_result_msg.detections.append(detection)
    # seg_img = np.sum(seg_imgs, axis=0)
    seg_img = np.zeros((IMG.shape[0], IMG.shape[1]), dtype=np.uint8)
    if seg_imgs is not None:
        # seg = seg_img.astype(np.uint8)
        # seg = seg * 128
        # if padding[1] == 0:
        #     if padding[0] != 0:
        #         seg2 = seg[:, padding[0]:-padding[2]]
        #     else:
        #         seg2 = seg
        # else:
        #     if padding[0] == 0:
        #         seg2 = seg[padding[1]:-padding[3], :]
        #     else:
        #         seg2 = seg[padding[1]:-padding[3], padding[0]:-padding[2]]

        # seg_img = cv2.resize(seg2, (IMG.shape[1], IMG.shape[0]), interpolation=cv2.INTER_LINEAR)
        for i in range(len(seg_imgs)):
            mask_img = seg_imgs[i]
            mask_img = mask_img.astype(np.uint8)
            mask_img = mask_img * 128
            if padding[1] == 0:
                if padding[0] != 0:
                    mask_img = mask_img[:, padding[0]:-padding[2]]
                else:
                    pass
            else:
                if padding[0] == 0:
                    mask_img = mask_img[padding[1]:-padding[3], :]
                else:
                    mask_img = mask_img[padding[1]:-padding[3], padding[0]:-padding[2]]
            mask_img = cv2.resize(mask_img, (IMG.shape[1], IMG.shape[0]), interpolation=cv2.INTER_LINEAR)
            seg_img = cv2.max(seg_img, mask_img)
            # seg = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            # IMG = cv2.add(IMG, seg)
            
            top_left_x, top_left_y, right_bottom_x,right_bottom_y = boxes[i]
            #恢复原图尺寸
            top_left_x = (top_left_x - padding[0])/ratio[0]
            top_left_y = (top_left_y - padding[1])/ratio[1]
            right_bottom_x = (right_bottom_x - padding[0])/ratio[0]
            right_bottom_y = (right_bottom_y - padding[1])/ratio[1]
            #限制范围不要超出原图大小
            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            right_bottom_x = min(mask_img.shape[1], right_bottom_x)
            right_bottom_y = min(mask_img.shape[0], right_bottom_y)
            
            crop_seg_img = mask_img[int(top_left_y):int(right_bottom_y), int(top_left_x):int(right_bottom_x)]
            mask_image_msg = bridge.cv2_to_imgmsg(
                crop_seg_img, encoding="mono8"
            )
            yolo_result_msg.masks.append(mask_image_msg)
    if boxes is not None and Draw_flag:
        # IMG = merge_seg2(IMG, seg_img)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
        IMG = cv2.add(IMG, seg_img)
        draw(IMG, boxes, scores, classes, ratio, padding)


    return yolo_result_msg, IMG
