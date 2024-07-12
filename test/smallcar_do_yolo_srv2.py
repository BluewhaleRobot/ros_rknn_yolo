import rospy
import cv2
from cv_bridge import CvBridge
from ros_rknn_yolo.srv import DoYolo, DoYoloRequest
from sensor_msgs.msg import Image
import signal
import time

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

    
def visualize_image(cv_image,name):
    cv2.imshow(name, cv_image)
    while rospy.is_shutdown() == False:
        if cv2.waitKey(1) != -1:
            break
        time.sleep(0.1)
        cv2.imshow(name, cv_image)

    cv2.destroyAllWindows()

def main():
    rospy.init_node('yolo_srv_test_client')

    try:
        rospy.wait_for_service('/yolo_node_2/do_yolo', timeout=5)
    except rospy.ROSException as e:
        print("Service did not become available before timeout.")
        return
        
    try:
        do_yolo = rospy.ServiceProxy('/yolo_node_2/do_yolo', DoYolo)
        bridge = CvBridge()

        # Load image using OpenCV
        cv_image = cv2.imread('../data/small-vehicles.jpg')
        # cv_image = cv2.imread('../data/bus.jpg')

        # Convert the image to ROS image message
        print("read image " + str(cv_image.shape))
        img_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        
        # Create a service request
        print("call service")
        req = DoYoloRequest()
        req.input_img = img_msg
        req.enable_crop = True
        req.enable_draw = True
        
        # Change the behavior of SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)

        # Change the behavior of SIGALRM
        signal.alarm(10)  # Ten seconds

        # Call the service
        try:
            # Call the service
            res = do_yolo(req)
        except TimeoutException:
            print("do_yolo service call timed out")
        else:
            # Reset the alarm
            signal.alarm(0)
            
        print("service done " + str(res.result) + " " + str(len(res.yolo_result.detections)) + " " + str(len(res.yolo_result.masks)))
        
        # Convert the response images to OpenCV images and visualize them
        yolo_result_img = bridge.imgmsg_to_cv2(res.yolo_result_img, "bgr8")
        visualize_image(yolo_result_img, "yolo_result_img")
        
        obj_index = 0
        for detection in res.yolo_result.detections:
            source_img = bridge.imgmsg_to_cv2(detection.source_img, "bgr8")

            if res.yolo_result.masks:
                mask = res.yolo_result.masks[obj_index]
                mask_img = bridge.imgmsg_to_cv2(mask, "mono8")
                #add to source_img
                source_img = cv2.add(source_img, cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR))
                
            visualize_image(source_img, "the "+str(obj_index)+" object_crop_img: " + detection.object_name)
            obj_index += 1
        
        #绘制masks

            
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == "__main__":
    main()