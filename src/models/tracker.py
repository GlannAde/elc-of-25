import math
import time
from enum import IntEnum
from models.Kalman import KalmanFilter3D

import cv2
import numpy as np
import yaml

from models.Kalman import KalmanFilter3D
from models.dm_serial import DM_Serial


class Status(IntEnum):
    LOST = 0
    TMP_LOST = 2
    TRACK = 3


class Tracker:
    def __init__(self, real_width=21.0, real_height=17.5, use_kf=True, imu_port=None, imu_baud=921600 ,imu_fusion_alpha=0.98):
        """
        :param real_width: 靶纸物理宽度 (单位: 厘米)
        :param real_height: 靶纸物理高度 (单位: 厘米)
        """
        # 将厘米转换为标准单位：米 (自瞄建议统一使用三维国际单位制)
        self.real_width = real_width / 100.0
        self.real_height = real_height / 100.0

        # 相机坐标系轴向偏置补偿 (米)
        self.ref_point = np.array([-0.08, 0.0, 0.0])
        self.yaw_bias = -2.3
        self.pitch_bias = -0.2

        self.use_kf = use_kf

        # --- 新增 IMU 融合参数 ---
        self.imu_fusion_alpha = imu_fusion_alpha   # 互补滤波系数 (0~1)，越大越信任 IMU
        self.imu_yaw = None                        # 最近一次 IMU 绝对 Yaw (度)
        self.imu_yaw_valid = False                 # IMU 数据是否有效
        self.last_imu_yaw_deg = 0.0                # 上一帧 IMU Yaw

        self.status = Status.LOST
        self.lost_count = 0
        self.frame_lost_tol = 8
        self.last_time = None

        self.onfire = False
        self.fire_deadzone = 1.5  # 允许开火的角度误差角度

        self.imu_serial = None
        if imu_port:
            self._init_imu(imu_port, imu_baud)

        if self.use_kf:
            self.kf = KalmanFilter3D()

        # 1. 定义三维物理模型点 (严格对应 detector.py 的 [tl, bl, br, tr] 顺序)
        W = self.real_width
        H = self.real_height
        self.object_points = np.array(
            [
                [-W / 2, -H / 2, 0.0],  # 左上 tl
                [-W / 2, H / 2, 0.0],  # 左下 bl
                [W / 2, H / 2, 0.0],  # 右下 br
                [W / 2, -H / 2, 0.0],  # 右上 tr
            ],
            dtype=np.float32,
        )

        # 2. 载入相机内参 (此处使用你的近似参数，标定后请替换)
        try:
            self.camera_matrix, self.dist_coeffs = self.load_camera_params(
                "config/camera_params.yaml"
            )
            print("成功加载相机 YAML 配置文件！")
        except FileNotFoundError:
            print("[警告] 未找到 camera_params.yaml，使用默认近似内参！")
            self.camera_matrix = np.array(
                [[725.6, 0.0, 320.0], [0.0, 725.6, 240.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def _init_imu(self, port, baud):
        """初始化 DM_Serial 并启动后台读线程"""
        try:
            self.imu_serial = DM_Serial(port, baud)
            if not self.imu_serial.is_open:
                print(f"[Tracker] IMU 打开失败: {self.imu_serial.last_error()}")
                return
            # 启动后台读取线程（不打印，只刷新最新帧）
            self.imu_serial.start_reader(read_sleep=0.001)
            # 稍等几帧，确保有数据
            time.sleep(0.1)
            # 获取一次初始 Yaw
            pkt, ts, cnt = self.imu_serial.get_latest()
            if pkt:
                rid, (ax, ay, az) = pkt   # 注意: DM_Serial 返回 (rid, (f1,f2,f3))
                # 这里假设达妙 IMU 数据格式为: [roll, pitch, yaw] 或 [acc_x, acc_y, acc_z]?
                # 实际需要根据你 IMU 的 RID 解读。常见的 RID=0x01 可能是欧拉角。
                # 默认第 3 个值为 Yaw
                self.imu_yaw = f3    # 单位度  # noqa: F821
                self.imu_yaw_valid = True
            print(f"[Tracker] IMU 已启动，初始 Yaw = {self.imu_yaw:.2f}°")
        except Exception as e:
            print(f"[Tracker] IMU 初始化失败: {e}")
            self.imu_serial = None

    def _get_imu_yaw(self):
        """从最新 IMU 帧中提取 Yaw 角（度），若无效返回 None"""
        if not self.imu_serial:
            return None
        pkt, ts, cnt = self.imu_serial.get_latest()
        if pkt:
            rid, (v1, v2, v3) = pkt
            # --- 重要：请根据你 IMU 的实际输出修改这里 ---
            # 如果 RID=0x01 对应 [Roll, Pitch, Yaw]，则 yaw = v3
            # 如果对应 [AccX, AccY, AccZ]，则需要额外积分得到 Yaw。
            # 以下假设 v3 就是 Yaw（度）
            yaw_deg = v3
            self.imu_yaw = yaw_deg
            self.imu_yaw_valid = True
            return yaw_deg
        return None

    def time_diff(self):
        # 计算时间间隔
        current_time = time.time_ns()
        if self.last_time is None:
            self.last_time = current_time
            return 1 / 120.0
        diff = (current_time - self.last_time) / 1e9
        self.last_time = current_time
        return min(diff, 0.1)

    def load_camera_params(self, yaml_path):
        """加载标准的 ROS 格式相机标定 YAML 文件"""
        import numpy as np

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # 读取 3x3 相机内参矩阵
        camera_matrix = np.array(
            config["camera_matrix"]["data"], dtype=np.float32
        ).reshape(config["camera_matrix"]["rows"], config["camera_matrix"]["cols"])

        # 读取 1x5 畸变系数 (注意这里的键名改成了 distortion_coefficients)
        dist_coeffs = np.array(
            config["distortion_coefficients"]["data"], dtype=np.float32
        ).reshape(
            config["distortion_coefficients"]["rows"],
            config["distortion_coefficients"]["cols"],
        )

        return camera_matrix, dist_coeffs

    def solve_pnp(self, board):
        """核心：通过 PnP 算法获取目标的 3D 物理空间坐标"""
        image_points = np.array(board.points, dtype=np.float32)

        # 使用专为平面 4 点定制的 IPPE 算法
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if success:
            # tvec 包含了相机坐标系下的 [X, Y, Z] (单位: 米)
            return True, tvec[0][0], tvec[1][0], tvec[2][0]
        return False, 0.0, 0.0, 0.0

    def filter_and_predict(self, target):
        dt = self.time_diff()

        # 解析当前的真实物理三维坐标
        success = False
        raw_x, raw_y, raw_z = 0.0, 0.0, 0.0
        if target and target.points:
            success, raw_x, raw_y, raw_z = self.solve_pnp(target)

        # 分支1：不开启卡尔曼滤波
        if not self.use_kf:
            if success:
                self.status = Status.TRACK
                return raw_x, raw_y, raw_z
            else:
                self.status = Status.LOST
                return 0.0, 0.0, 0.1

        # 分支2：开启卡尔曼滤波
        if success:
            if self.status == Status.LOST:
                self.kf.reset(raw_x, raw_y, raw_z)  # 丢失重捕【热启动】
            self.status = Status.TRACK
            self.lost_count = 0

            self.kf.predict(dt=dt)
            return self.kf.update(raw_x, raw_y, raw_z)
        else:
            self.lost_count += 1
            if self.lost_count <= self.frame_lost_tol:
                self.status = Status.TMP_LOST
                return self.kf.predict(dt=dt)  # 靠3D惯性继续预测
            else:
                self.status = Status.LOST
                self.kf.reset()
                return 0.0, 0.0, 0.1

    def project_3d_to_2d(self, x, y, z):
        """工具函数：将三维物理坐标重新投影回二维屏幕（用于画瞄准十字）"""
        pt_3d = np.array([[x, y, z]], dtype=np.float32)
        rvec_zero = np.zeros((3, 1), dtype=np.float32)
        tvec_zero = np.zeros((3, 1), dtype=np.float32)
        img_pts, _ = cv2.projectPoints(
            pt_3d, rvec_zero, tvec_zero, self.camera_matrix, self.dist_coeffs
        )
        return int(img_pts[0][0][0]), int(img_pts[0][0][1])

    def track(self, board, mode="TRACK", real_radius_m=0.15, period_sec=3.0):
        fx, fy, fz = self.filter_and_predict(board)

        if self.status != Status.LOST:
            vx = self.kf.kf.statePost[3,0] if self.use_kf else 0.0
            vy = self.kf.kf.statePost[4,0] if self.use_kf else 0.0
            vz = self.kf.kf.statePost[5,0] if self.use_kf else 0.0
            sys_delay = 0.05
            aim_x = fx + vx * sys_delay
            aim_y = fy + vy * sys_delay
            aim_z = fz + vz * sys_delay

            if mode == "CIRCLE":
                t = time.time()
                omega = 2 * math.pi / period_sec
                aim_x += real_radius_m * math.cos(omega * t)
                aim_y += real_radius_m * math.sin(omega * t)

            # ----- 视觉计算原始 Yaw/Pitch -----
            yaw_err_vision = -math.degrees(math.atan2(aim_x + self.ref_point[0], aim_z + self.ref_point[2]))
            horizontal_dist = math.hypot(aim_x, aim_z)
            pitch_err_vision = math.degrees(math.atan2(aim_y + self.ref_point[1], max(horizontal_dist, 0.1)))

            # ----- 融合 IMU Yaw -----
            imu_yaw = self._get_imu_yaw()
            if imu_yaw is not None and self.imu_yaw_valid:
                # 简单的互补滤波，对 yaw_err 做平滑（系数可调）
                if not hasattr(self, '_filtered_yaw'):
                    self._filtered_yaw = yaw_err_vision
                self._filtered_yaw = self.imu_fusion_alpha * self._filtered_yaw + (1 - self.imu_fusion_alpha) * yaw_err_vision
                yaw_err = self._filtered_yaw
                # 可选：pitch 也可以融合 IMU pitch
                # pitch_err = ...
            else:
                yaw_err = yaw_err_vision
                self._filtered_yaw = yaw_err_vision

            # 添加 bias 补偿（仍保留原有）
            yaw_err += self.yaw_bias
            pitch_err_vision += self.pitch_bias
            # 注意：pitch 未融合 IMU，直接使用视觉结果
            pitch_err = pitch_err_vision

            self.onfire = (self.status == Status.TRACK and abs(yaw_err) < self.fire_deadzone and abs(pitch_err) < self.fire_deadzone)

            # 投影 2D 点等...
            smooth_center_2d = self.project_3d_to_2d(fx, fy, fz)
            aim_point_2d = self.project_3d_to_2d(aim_x, aim_y, aim_z)
            laser_pos_2d = self.project_3d_to_2d(0.0, 0.0, fz)
            return (yaw_err, pitch_err, fz, self.status, laser_pos_2d, smooth_center_2d, aim_point_2d)
        else:
            self.onfire = False
            return 0.0, 0.0, 0.0, self.status, None, None, None
