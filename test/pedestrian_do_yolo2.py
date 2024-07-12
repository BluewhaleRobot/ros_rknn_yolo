import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from dynamic_reconfigure.client import Client
Bridge = CvBridge()

def image_callback(msg):
    try:
        cv_image = Bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(e)
    else:
        cv2.imshow("YOLO_NODE_2 Result Image", cv_image)
        cv2.waitKey(1)

def main():
    rospy.init_node('image_publisher2', anonymous=True)
    try:
        #通过dynamic reconfigure设置yolo_node_1的参数，/yolo_node_1/drop_interval /yolo_node_1/enable_crop /yolo_node_1/enable_draw /yolo_node_1/enable_streaming_detect
        # 创建dynamic_reconfigure的客户端
        client = Client("/yolo_node_2",timeout=5)

        # 设置参数
        params = {
            'enable_crop': True,
            'enable_draw': True,
            'enable_streaming_detect': True,
            'drop_interval': 1,
            'raw_img_topic_name': '/camera_node2/image_raw'
        }
        # 更新参数
        client.update_configuration(params)
    except Exception as e:
        print(e)
        return
                    
    pub = rospy.Publisher('/camera_node2/image_raw', Image, queue_size=10)
    rospy.Subscriber('/yolo_node_2/yolo_result_img', Image, image_callback)
    cap = cv2.VideoCapture('../data/pedestrian.mp4')
    bridge = CvBridge()

    rate = rospy.Rate(30) # 30hz
    while cap.isOpened() and not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            timestamp = rospy.Time.now()
            img_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
            img_msg.header.stamp = timestamp
            pub.publish(img_msg)
        else:
            break
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass