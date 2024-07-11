#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR
source ../venv/bin/activate
cd ~/.ros
$DIR/../venv/bin/python $DIR/rknn_yolo_node.py $@