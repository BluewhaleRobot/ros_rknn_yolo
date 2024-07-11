cmake_minimum_required(VERSION 3.14.1)
project(ros_rknn_yolo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
  cv_bridge
  dynamic_reconfigure
  vision_msgs
)

find_package(OpenCV REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/RknnYolo.cfg
)

add_service_files(
   FILES
   DoYolo.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   vision_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_generation cv_bridge message_runtime vision_msgs
)

# update python dependencies
execute_process(COMMAND "virtualenv" "--system-site-packages" "-p" "/usr/bin/python" "${PROJECT_SOURCE_DIR}/venv")
message("Installing rknn_toolkit_lite2 dependencies of ${PROJECT_NAME}")
execute_process(
  COMMAND "${PROJECT_SOURCE_DIR}/venv/bin/pip" "install" ${PROJECT_SOURCE_DIR}/rknn2_runtime/rknn_toolkit_lite2-2.0.0b1-cp310-cp310-linux_aarch64.whl
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  RESULT_VARIABLE result
  ERROR_VARIABLE error_var
  OUTPUT_VARIABLE out_var
)

if("${result}" STREQUAL "0")
  message("Install rknn_toolkit_lite2  success")
else()
  message(FATAL_ERROR "Install rknn_toolkit_lite2  failed")
endif()

message("Installing python dependencies of ${PROJECT_NAME}")
execute_process(
  COMMAND "${PROJECT_SOURCE_DIR}/venv/bin/pip" "install"  "-i" "https://pypi.mirrors.ustc.edu.cn/simple" "--timeout" "120" "-r" "${PROJECT_SOURCE_DIR}/requirements.txt"
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  RESULT_VARIABLE result
  ERROR_VARIABLE error_var
  OUTPUT_VARIABLE out_var
)
message("${PROJECT_SOURCE_DIR}/venv/bin/pip ${PROJECT_SOURCE_DIR}/requirements.txt")
if("${result}" STREQUAL "0")
  message("Install python dependencies success")
else()
  message(FATAL_ERROR "Install python dependencies failed")
endif()