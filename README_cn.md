# ROS节点：rknn_yolo_node

## 概述
`rknn_yolo_node`是一个ROS节点，使用RKNN（Rockchip NPU神经网络API）模型进行对象检测。它订阅一个图像话题，使用YOLO（You Only Look Once）对象检测算法处理图像，并发布检测结果。

## 订阅的话题
- `raw_img_topic_name` (`sensor_msgs/Image`): 此话题可以动态配置。节点订阅此话题以接收用于对象检测的原始图像。默认话题名称是`/camera_node/image_raw`。

## 发布的话题
- `yolo_output_msg` (`vision_msgs/YoloResult`): 此话题发布YOLO对象检测的结果。每条消息包括检测到的对象及其边界框。
- `yolo_output_img_msg` (`sensor_msgs/Image`): 如果`enable_draw`参数设置为`True`，此话题将发布带有突出显示的检测对象的图像。

## 服务
- `~do_yolo` (`ros_rknn_yolo/DoYolo`): 此服务在提供的图像上执行YOLO对象检测，并返回检测结果。

## 参数
- `~rknn_model_function` (字符串，默认值: 'yolov8_func'): RKNN模型后处理函数文件的名称。
- `~rknn_model_path` (字符串，默认值: '/home/xiaoqiang/Documents/ros/src/ros_rknn_yolo/model/yolov8s.rknn'): RKNN模型文件的路径。
- `~enable_streaming_detect` (布尔值，默认值: `False`): 如果为`True`，节点将连续处理来自`raw_img_topic_name`话题的图像。
- `~enable_crop` (布尔值，默认值: `False`): 如果为`True`，对象检测结果将包含其裁剪图像。
- `~enable_draw` (布尔值，默认值: `False`): 如果为`True`，节点将发布带有突出显示的检测对象的图像到`yolo_output_img_msg`话题。
- `~drop_interval` (整数，默认值: 1): 丢弃帧的间隔，用于控制处理速度。
- `~obj_thresh` (浮点数，默认值: 0.25): 对象检测的置信度阈值。
- `~nms_thresh` (浮点数，默认值: 0.45): 对象检测中非最大抑制的阈值。
- `~classes` (字符串列表): 模型可以检测的对象类别的列表。
- `~tpes` (整数，默认值: 1): 用于并行处理的线程池执行器的数量。
- `~npu_start_id` (整数，默认值: 0): NPU（神经处理单元）的起始ID。
- `~hide_label` (布尔值，默认值: `False`): 如果为`True`，节点将不会发布带有检测对象名称的图像。

## 动态配置参数
- `enable_crop` (布尔值): 启用或禁用对象检测结果图像裁剪。
- `enable_draw` (布尔值): 启用或禁用在发布的图像中突出显示检测到的对象。
- `enable_streaming_detect` (布尔值): 启用或禁用连续图像处理。
- `drop_interval` (整数): 设置丢弃帧的间隔，以控制处理速度。
- `raw_img_topic_name` (字符串): 设置要订阅的图像话题的名称。

## 安装说明

git克隆然后catkin_make, 依赖的vision_msgs包在这里：http://git.bwbot.org/publish/vision_msgs.git

### Post-catkin-make-编译说明

使用catkin编译后，您需要执行`sudo ./fix_rknn2_runtime`来更新rknn2_runtime，以确保npu库函数版本的一致性。

## 启动说明

`xiaoqiang_yolo.launch`是一个参考文件，它启动了两个`rknn_yolo_node`节点，名为`yolo_node_1`和`yolo_node_2`。这两个节点都使用`ros_rknn_yolo`包中的`rknn_yolo_node.sh`脚本。

请注意，`rknn_yolo_node.sh`脚本在Python虚拟环境中运行。这是因为我们的`rknn_yolo_node`节点依赖于特定版本的Python库，这些库安装在虚拟环境中。因此，当我们启动`yolo_node_1`和`yolo_node_2`节点时，我们实际上是在Python虚拟环境中启动`rknn_yolo_node.sh`脚本，而不是直接启动`rknn_yolo_node`。

### 关于启动文件

每个节点在启动时加载一个默认的参数配置文件，位于`ros_rknn_yolo`包的`config`目录中。对于`yolo_node_1`，默认配置文件是`default.yaml`，对于`yolo_node_2`，它是`default2.yaml`。

如果`use_custom_setting`参数设置为`true`，每个节点还加载一个额外的参数配置文件。对于`yolo_node_1`，这个文件是`do_yolo_1.yaml`，对于`yolo_node_2`，它是`do_yolo_2.yaml`。这些文件位于`startup`包的`params`目录中。

### 话题重映射

每个节点发布两个话题：`yolo_output_msg`和`yolo_output_img_msg`。这些话题的名称在启动文件中被重映射。对于`yolo_node_1`，`yolo_output_msg`被重映射为`/yolo_node_1/yolo_result`，`yolo_output_img_msg`被重映射为`/yolo_node_1/yolo_result_img`。对于`yolo_node_2`，`yolo_output_msg`被重映射为`/yolo_node_2/yolo_result`，`yolo_output_img_msg`被重映射为`/yolo_node_2/yolo_result_img`。

### 运行启动文件

要运行此启动文件，您可以在终端中输入以下命令：

```bash
roslaunch xiaoqiang_yolo xiaoqiang_yolo.launch
```

如果您想使用自定义参数配置，可以在运行命令时添加`use_custom_setting:=true`：

```bash
roslaunch xiaoqiang_yolo xiaoqiang_yolo.launch use_custom_setting:=true
```

### 运行基本测试

要运行基本测试，请按照以下步骤操作：

1. 导航到`ros_rknn_yolo`包的测试目录：

```bash
roscd ros_rknn_yolo/test
```

2. 运行`bus_do_yolo_srv.py`脚本。此脚本向`~do_yolo`服务发送一个服务请求，触发YOLO对象检测过程：

```bash
python bus_do_yolo_srv.py
```

3. 观察输出。脚本将把检测结果打印到控制台。每个结果包括检测对象的类别和边界框。

请注意，`bus_do_yolo_srv.py`脚本需要一个正在运行的`rknn_yolo_node`实例。在运行测试之前，请确保启动节点：

```bash
roslaunch xiaoqiang_yolo xiaoqiang_yolo.launch
```