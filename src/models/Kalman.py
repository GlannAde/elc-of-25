import cv2
import numpy as np


class KalmanFilter3D:

    def __init__(self, q_scale=0.2, r_xyz_scale=0.05, default_dt=1 / 120.0):
        self.dt = default_dt
        # 6维状态量: [X, Y, Z, VX, VY, VZ] -> 均为物理三维空间量 (单位: 米)
        # 3维观测量: [X, Y, Z] -> PnP 直接输出的物理三维坐标
        self.kf = cv2.KalmanFilter(6, 3)

        # 状态转移矩阵 (F)
        # --- [優化] 引入 0.6 的速度阻尼，防止雲台轉動產生的虛假觀測速度無限累積 ---
        self.damping = 0.6
        self.max_velocity = 4.0  # 限制目標在物理世界中的最大可能物理速度 (米/秒)

        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, self.damping, 0, 0],  
                [0, 0, 0, 0, self.damping, 0],
                [0, 0, 0, 0, 0, self.damping],
            ],
            np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # 过程噪声协方差矩阵 (Q)
        self.kf.processNoiseCov = (
            np.array(
                [
                    [0.05, 0, 0, 0, 0, 0],
                    [0, 0.05, 0, 0, 0, 0],
                    [0, 0, 0.05, 0, 0, 0],
                    [0, 0, 0, 0.4, 0, 0],
                    [0, 0, 0, 0, 0.4, 0],
                    [0, 0, 0, 0, 0, 0.4],
                ],
                dtype=np.float32,
            )
            * q_scale
        )

        self.base_q = self.kf.processNoiseCov.copy()
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * r_xyz_scale
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 100.0

    def reset(self, init_x=0.0, init_y=0.0, init_z=0.0):
        """丢靶后重新捕获时的【热启动】"""
        self.kf.statePost = np.array(
            [[init_x], [init_y], [init_z], [0.0], [0.0], [0.0]], np.float32
        )
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 100.0

    def predict(self, dt=None):
        if dt is not None and dt > 0:
            self.dt = dt
            self.kf.transitionMatrix[0, 3] = dt
            self.kf.transitionMatrix[1, 4] = dt
            self.kf.transitionMatrix[2, 5] = dt
            self.kf.processNoiseCov = self.base_q * (dt / (1 / 120.0))
        prediction = self.kf.predict()
        return prediction[0, 0], prediction[1, 0], prediction[2, 0]

    def update(self, x, y, z):
        measure = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
        estimate = self.kf.correct(measure)

        # --- [優化] 速度防線：對正確後的狀態速度進行硬上限裁剪，徹底杜絕數據飛車 ---
        self.kf.statePost[3, 0] = np.clip(
            self.kf.statePost[3, 0], -self.max_velocity, self.max_velocity
        )
        self.kf.statePost[4, 0] = np.clip(
            self.kf.statePost[4, 0], -self.max_velocity, self.max_velocity
        )
        self.kf.statePost[5, 0] = np.clip(
            self.kf.statePost[5, 0], -self.max_velocity, self.max_velocity
        )

        return estimate[0, 0], estimate[1, 0], estimate[2, 0]
