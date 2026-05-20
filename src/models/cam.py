import queue
import threading
import time

import cv2


class Camera:
    def __init__(self, index=4, width=640, height=480, format="MJPG", fps=120):
        """
        基于长度为1的队列优化的相机类
        """
        self.index = self.find_index(index)
        if self.index is None:
            raise RuntimeError("无法找到可用的摄像头索引")

        self.cam = cv2.VideoCapture(self.index, cv2.CAP_V4L2)

        # 强制设置硬件解码格式和高帧率
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format))
        self.cam.set(cv2.CAP_PROP_FPS, fps)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cam.get(cv2.CAP_PROP_FPS)

        # 核心修改：初始化长度为 1 的队列
        self.q = queue.Queue(maxsize=1)
        self.running = True

        # 开启守护线程
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

        print(
            f"Camera Initialized: Index={self.index}, Size={self.width}x{self.height}, Target FPS={fps}, Actual FPS={self.actual_fps}"
        )

    def _update(self):
        """生产者线程：疯狂采图，只留最新"""
        while self.running:
            ret, frame = self.cam.read()
            if ret:
                # 如果队列满了，主动丢弃旧帧腾出位置
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass

                # 放入最新帧
                self.q.put((ret, frame))
            else:
                time.sleep(0.01)

    def read(self):
        """
        消费者方法：主线程调用此方法获取最新帧
        """
        try:
            # 阻塞等待最新帧，最多等 1 秒防止程序死锁
            ret, frame = self.q.get(timeout=1.0)
            return ret, frame
        except queue.Empty:
            return False, None

    def find_index(self, start_index=0):
        for i in range(start_index, start_index + 10):
            temp_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if temp_cap.isOpened():
                temp_cap.release()
                return i
        return None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cam.release()
        print("Camera source released.")


if __name__ == "__main__":
    try:
        my_cam = Camera(index=0, width=640, height=480, fps=120)
        while True:
            r, f = my_cam.read()
            if r:
                cv2.imshow("Queue Cam Test", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        my_cam.release()
        cv2.destroyAllWindows()
