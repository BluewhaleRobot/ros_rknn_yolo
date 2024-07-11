from queue import Queue
import queue
import time
# import torch
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
import cv_bridge

def initRKNN(rknnModel="../model/yolov8s.rknn", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="../model/yolov8s.rknn", TPEs=1, NpuStartId=0):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3 + NpuStartId))
    return rknn_list


class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, NpuStartId, func):
        self.TPEs = TPEs
        self.queue = {}
        self.queue_lock = threading.Lock()
        self.result_queue = Queue()  # 新增一个队列用于存储结果
        self.rknnPool = initRKNNs(rknnModel, TPEs, NpuStartId)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0
        self.bridge = cv_bridge.CvBridge()

    def put(self, frame, header, crop_object_flag, draw_flag):
        fut = self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], self.bridge, frame, header, crop_object_flag, draw_flag)
        fut.add_done_callback(self._on_done)  # 当任务完成时，调用_on_done方法
        with self.queue_lock:
            self.queue[id(fut)] = fut
        self.num += 1

    def _on_done(self, fut: Future):
        # 当任务完成时，将结果添加到结果队列中
        self.result_queue.put(fut.result())
        with self.queue_lock:
            del self.queue[id(fut)]

    def get(self):
        if self.result_queue.empty():
            return None, False
        result = self.result_queue.get()
        return result, True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
            
    def clearqueue(self):
        # Shutdown the ThreadPoolExecutor, it will block until all tasks are done
        is_empty_now = False
        while not is_empty_now:
            with self.queue_lock:
                is_empty_now =  (len(self.queue) == 0)
            time.sleep(0.05)
            
        while not self.result_queue.empty():  # 清空结果队列
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.05)
        self.num = 0
        
    def getQueueSize(self):
        with self.queue_lock:
            return len(self.queue)