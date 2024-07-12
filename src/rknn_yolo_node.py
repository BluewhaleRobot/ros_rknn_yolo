import importlib
from rknnpool import rknnPoolExecutor
from dynamic_reconfigure.server import Server
from ros_rknn_yolo.cfg import RknnYoloConfig
import cv_bridge
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D,Detection2DArray,ObjectHypothesisWithPose,YoloResult
from sensor_msgs.msg import Image

import threading
from ros_rknn_yolo.srv import DoYolo, DoYoloResponse
import rospy

ROS_NODE = rospy.init_node('~')

RKNN_MODEL_FUNCTION_FILE = rospy.get_param('~rknn_model_function', 'yolov8_func')
RKNN_MODEL_FUNCTION = importlib.import_module(RKNN_MODEL_FUNCTION_FILE)
RKNN_MODEL_PATH = rospy.get_param('~rknn_model_path', '/home/xiaoqiang/Documents/ros/src/ros_rknn_yolo/model/yolov8s.rknn')

ENABLE_STREAMING_DETECT = rospy.get_param('~enable_streaming_detect', False)
ENABLE_CROP = rospy.get_param('~enable_crop', False)
ENABLE_DRAW = rospy.get_param('~enable_draw', False)
RAW_IMG_TOPIC_NAME = rospy.get_param('~raw_img_topic_name', '/camera/color/image_raw')
DROP_INTERVAL = rospy.get_param('~drop_interval', 1)
OBJ_THRESH = rospy.get_param('~obj_thresh', 0.25)
NMS_THRESH = rospy.get_param('~nms_thresh', 0.45)
CLASSES = rospy.get_param('~classes', ["person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "])
TPES = rospy.get_param('~tpes', 1)
NPU_START_ID = rospy.get_param('~npu_start_id', 0)

# from rknnpool import rknnPoolExecutor

POOL = rknnPoolExecutor(
    rknnModel=RKNN_MODEL_PATH,
    TPEs=TPES,
    NpuStartId=NPU_START_ID,
    func=RKNN_MODEL_FUNCTION.rknn_Func)
# POOL = None

CFG_LOCK = threading.Lock()
POOL_LOCK1 = threading.Lock()
POOL_LOCK2 = threading.Lock()

class YoloSubscriber:
    def __init__(self):
        self.subscriber = None
        self.bridge = cv_bridge.CvBridge()
        self.subscriber_topic_name = None 
        self.server = Server(RknnYoloConfig, self.cfg_callback)
        self.yolo_result_pub = rospy.Publisher('yolo_output_msg', YoloResult, queue_size=10)
        self.yolo_result_img_pub = rospy.Publisher('yolo_output_img_msg', Image, queue_size=10)

    def cfg_callback(self, config, level):
        with CFG_LOCK:
            global ENABLE_CROP
            global ENABLE_DRAW
            global ENABLE_STREAMING_DETECT
            global DROP_INTERVAL
            ENABLE_CROP = config.enable_crop
            ENABLE_DRAW = config.enable_draw
            ENABLE_STREAMING_DETECT = config.enable_streaming_detect
            DROP_INTERVAL = config.drop_interval

        if config.enable_streaming_detect:
            if not self.subscriber:
                self.subscriber = rospy.Subscriber(config.raw_img_topic_name, Image, self.image_callback, queue_size=2)
                self.subscriber_topic_name = config.raw_img_topic_name
            else:
                if self.subscriber_topic_name != config.raw_img_topic_name:
                    self.subscriber.unregister()
                    self.subscriber = rospy.Subscriber(config.raw_img_topic_name, Image, self.image_callback, queue_size=2)
                    self.subscriber_topic_name = config.raw_img_topic_name
        else:
            if self.subscriber is not None:
                self.subscriber.unregister()
                self.subscriber = None
                self.subscriber_topic_name = None
        
        return config
        
    def image_callback(self, img_msg):
        #将data转换成opencv格式然后传入rknn模型进行检测
        #get lock
        with CFG_LOCK:
            enable_crop = ENABLE_CROP
            enable_draw = ENABLE_DRAW
            enable_streaming_detect = ENABLE_STREAMING_DETECT
            drop_interval = DROP_INTERVAL + 1
        #release the lock
        if enable_streaming_detect:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            header = img_msg.header
            if header.seq % drop_interval == 0:
                with POOL_LOCK1:
                    if POOL.getQueueSize() >= TPES*2:
                        return
                    if cv_image is None or cv_image.shape[0] < 6 or cv_image.shape[1] < 6:
                        return
                    POOL.put(cv_image, header, enable_crop, enable_draw)
            else:
                pass
        else:
            pass
        
    def publish_msgs(self, event=None):
        #get lock
        with POOL_LOCK2:
            with CFG_LOCK:
                enable_draw = ENABLE_DRAW
                enable_streaming_detect = ENABLE_STREAMING_DETECT
            #release the lock
            #get the result from the rknn model
            if enable_streaming_detect:
                result, flag = POOL.get()
                if flag:
                    self.yolo_result_pub.publish(result[0])
                    if enable_draw:
                        img_msg = self.bridge.cv2_to_imgmsg(result[1], "bgr8")
                        img_msg.header = result[0].header
                        self.yolo_result_img_pub.publish(img_msg)
                else:
                    pass
            else:
                pass

    def do_yolo_srv(self, req):
        # print("do_yolo_srv1")
        with POOL_LOCK1:
            with POOL_LOCK2:
                # print("do_yolo_srv2")
                POOL.clearqueue()
                # print("do_yolo_srv3")
                cv_image = self.bridge.imgmsg_to_cv2(req.input_img, "bgr8")
                #check cv_image is None or too small size
                if cv_image is None or cv_image.shape[0] < 6 or cv_image.shape[1] < 6:
                    res = DoYoloResponse()
                    res.result = False
                    return res
                # print("do_yolo_srv4")
                header = Header()
                POOL.put(cv_image, header, req.enable_crop, req.enable_draw)
                # print("do_yolo_srv5")
                flag = False
                while POOL.getQueueSize() > 0 and rospy.is_shutdown() == False:
                    #sleep 10ms
                    rospy.sleep(0.01)
                # print("do_yolo_srv6")   
                while not flag and rospy.is_shutdown() == False:
                    result, flag = POOL.get()
                    #sleep 10ms
                    rospy.sleep(0.01)
                # print("do_yolo_srv7")
                res = DoYoloResponse()
                res.result = flag
                res.yolo_result = result[0]
                res.yolo_result_img = self.bridge.cv2_to_imgmsg(result[1], "bgr8")
                # print("do_yolo_srv8")
                return res
        
if __name__ == "__main__":
    ds = YoloSubscriber()
    #start the thread to publish the result
    rospy.Timer(rospy.Duration(0.02), ds.publish_msgs)
    rospy.Service('~do_yolo', DoYolo, ds.do_yolo_srv)
    rospy.spin()
    POOL.release()