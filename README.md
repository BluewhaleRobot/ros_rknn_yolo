# ROS Node: rknn_yolo_node

## Overview
The `rknn_yolo_node` is a ROS node that uses the RKNN (Rockchip NPU Neural Networks API) model for object detection. It subscribes to an image topic, processes the images using the YOLO (You Only Look Once) object detection algorithm, and publishes the detection results.

## Subscribed Topics
- `raw_img_topic_name` (`sensor_msgs/Image`): This topic is dynamically reconfigurable. The node subscribes to this topic to receive raw images for object detection. The default topic name is `/camera_node/image_raw`.

## Published Topics
- `yolo_output_msg` (`vision_msgs/YoloResult`): This topic publishes the results of the YOLO object detection. Each message includes the detected objects and their bounding boxes.
- `yolo_output_img_msg` (`sensor_msgs/Image`): This topic publishes the images with the detected objects highlighted, if the `enable_draw` parameter is set to `True`.

## Services
- `~do_yolo` (`ros_rknn_yolo/DoYolo`): This service performs YOLO object detection on a provided image and returns the detection results.

## Parameters
- `~rknn_model_function` (string, default: 'yolov8_func'): The name of the RKNN model post-process function file.
- `~rknn_model_path` (string, default: '/home/xiaoqiang/Documents/ros/src/ros_rknn_yolo/model/yolov8s.rknn'): The path to the RKNN model file.
- `~enable_streaming_detect` (bool, default: `False`): If `True`, the node will continuously process images from the `raw_img_topic_name` topic.
- `~enable_crop` (bool, default: `False`): If `True`, the objection detection result would contain it's crop image.
- `~enable_draw` (bool, default: `False`): If `True`, the node will publish images with the detected objects highlighted to the `yolo_output_img_msg` topic.
- `~drop_interval` (int, default: 1): The interval at which frames are dropped to control the processing speed.
- `~obj_thresh` (float, default: 0.25): The confidence threshold for object detection.
- `~nms_thresh` (float, default: 0.45): The threshold for non-maximum suppression in object detection.
- `~classes` (list of strings): The list of object classes that the model can detect.
- `~tpes` (int, default: 1): The number of thread pool executors for parallel processing.
- `~npu_start_id` (int, default: 0): The start ID for the NPU (Neural Processing Unit).
- `~hide_label` (bool, default: `False`): If `True`, the node will not publish images with the detected objects names.

## Dynamic Reconfigure Parameters
- `enable_crop` (bool): Enables or disables objection detection result image cropping.
- `enable_draw` (bool): Enables or disables the highlighting of detected objects in the published images.
- `enable_streaming_detect` (bool): Enables or disables continuous image processing.
- `drop_interval` (int): Sets the interval at which frames are dropped to control the processing speed.
- `raw_img_topic_name` (string): Sets the name of the image topic to subscribe to.

## Install Instructions

Git clone and then catkin_make, the dependent vision_msgs package is here: http://git.bwbot.org/publish/vision_msgs.git

### Post-catkin-make-Compilation Instructions

After compiling with catkin, you need to execute `sudo ./fix_rknn2_runtime` to update rknn2_runtime, so as to ensure the consistency of the npu library function version.

## Launch Instructions

The `xiaoqiang_yolo.launch` is a reference file, which launches two `rknn_yolo_node` nodes, named `yolo_node_1` and `yolo_node_2`. Both nodes use the `rknn_yolo_node.sh` script from the `ros_rknn_yolo` package.

Please note that the `rknn_yolo_node.sh` script is run within a Python virtual environment. This is because our `rknn_yolo_node` nodes depend on specific versions of Python libraries, which are installed in the virtual environment. Therefore, when we launch the `yolo_node_1` and `yolo_node_2` nodes, we are actually launching the `rknn_yolo_node.sh` script within the Python virtual environment, not the `rknn_yolo_node` directly.

### About The Launch File

Each node loads a default parameter configuration file upon launch, located in the `config` directory of the `ros_rknn_yolo` package. For `yolo_node_1`, the default configuration file is `default.yaml`, and for `yolo_node_2`, it is `default2.yaml`.

If the `use_custom_setting` argument is set to `true`, each node also loads an additional parameter configuration file. For `yolo_node_1`, this file is `do_yolo_1.yaml`, and for `yolo_node_2`, it is `do_yolo_2.yaml`. These files are located in the `params` directory of the `startup` package.

### Topic Remapping

Each node publishes two topics: `yolo_output_msg` and `yolo_output_img_msg`. The names of these topics are remapped in the launch file. For `yolo_node_1`, `yolo_output_msg` is remapped to `/yolo_node_1/yolo_result`, and `yolo_output_img_msg` is remapped to `/yolo_node_1/yolo_result_img`. For `yolo_node_2`, `yolo_output_msg` is remapped to `/yolo_node_2/yolo_result`, and `yolo_output_img_msg` is remapped to `/yolo_node_2/yolo_result_img`.

### Running the Launch File

To run this launch file, you can enter the following command in the terminal:

```bash
roslaunch xiaoqiang_yolo xiaoqiang_yolo.launch
```

If you want to use custom parameter configurations, you can add `use_custom_setting:=true` when running the command:

```bash
roslaunch xiaoqiang_yolo xiaoqiang_yolo.launch use_custom_setting:=true
```

### Running Basic Test

To run the basic test, follow these steps:

1. Navigate to the test directory of the `ros_rknn_yolo` package:

```bash
roscd ros_rknn_yolo/test
```

2. Run the `bus_do_yolo_srv.py` script. This script sends a service request to the `~do_yolo` service, which triggers the YOLO object detection process:

```bash
python bus_do_yolo_srv.py
```

3. Observe the output. The script will print the detection results to the console. Each result includes the class and bounding box of the detected object.

Please note that the `bus_do_yolo_srv.py` script requires a running instance of the `rknn_yolo_node`. Make sure to launch the node before running the test:

```bash
roslaunch xiaoqiang_yolo xiaoqiang_yolo.launch
```