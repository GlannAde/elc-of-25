import math
import time

import cv2

# --- 导入视觉与控制模块 ---
from models.cam import Camera
from models.detector import Detector
from models.pid import PIDController

# [硬件屏蔽] from models.status import GPIN
# --- 导入硬件驱动模块 ---
# [硬件屏蔽]
from models.stepper import EmmMotor
from models.tracker import Status, Tracker

# ==================== 系统参数配置区 ====================
CAMERA_INDEX = 0  # 摄像头索引 (如果在 Windows 测试，通常是 0 或 1)
PORT = "/dev/ttyACM0"
# PITCH_PORT = "/dev/ttyACM1"

USE_KF = True  # 是否启用 3D 卡尔曼滤波预测
SHOW_WINDOWS = True  # 是否显示调试画面和参数控制台 (设为 False 可榨干极限性能)

# 齿轮传动比配置
GEAR_RATIO_YAW = 5.09
GEAR_RATIO_PITCH = 9.23
# ========================================================

# ==================== 模块初始化 ====================
# 初始化相机和视觉算法模块
camera = Camera(index=CAMERA_INDEX, width=640, height=480, format="MJPG", fps=120)
detector = Detector(min_area=5000, max_area=500000, use_adaptive=True)

tracker = Tracker(
    real_width=21.0,
    real_height=17.5,
    use_kf=True,
    imu_port="/dev/ttyACM1",     # 你的 IMU 串口
    imu_baud=921600,
    imu_fusion_alpha=0.95
)

# [硬件屏蔽] 电机初始化
stepper_yaw = EmmMotor(port=PORT, baudrate=115200, timeout=1, motor_id=1)
stepper_pitch = EmmMotor(port=PORT, baudrate=115200, timeout=1, motor_id=2)

# PID 初始化 (保留 PID 对象用于在终端观察输出运算结果)
pid_yaw = PIDController(Kp=0.0, Ki=0.0, Kd=0.0, dt=1 / 120.0)
pid_pitch = PIDController(Kp=0.0, Ki=0.0, Kd=0.0, dt=1 / 120.0)

# [硬件屏蔽] GPIO 外设
# lazer = GPIN(pin=16, mode=1)
# heart_beat = GPIN(pin=18, mode=1)
# ========================================================


def nothing(x):
    pass


def init_board():
    """初始化调试窗口和 PID 动态调参滑动条"""
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 350)
    cv2.namedWindow("Tracker", cv2.WINDOW_FREERATIO)

    cv2.createTrackbar("yaw_kp", "Controls", 50, 1000, nothing)
    cv2.createTrackbar("yaw_ki", "Controls", 0, 1000, nothing)
    cv2.createTrackbar("yaw_kd", "Controls", 10, 1000, nothing)

    cv2.createTrackbar("pitch_kp", "Controls", 50, 1000, nothing)
    cv2.createTrackbar("pitch_ki", "Controls", 0, 1000, nothing)
    cv2.createTrackbar("pitch_kd", "Controls", 10, 1000, nothing)

    cv2.createTrackbar("vel_rpm", "Controls", 4000, 5000, nothing)
    cv2.createTrackbar("acc", "Controls", 200, 255, nothing)


def update_params():
    """读取滑块参数并实时更新给算法层"""
    yaw_kp = cv2.getTrackbarPos("yaw_kp", "Controls") / 1000.0
    yaw_ki = cv2.getTrackbarPos("yaw_ki", "Controls") / 100000.0
    yaw_kd = cv2.getTrackbarPos("yaw_kd", "Controls") / 100000.0

    pitch_kp = cv2.getTrackbarPos("pitch_kp", "Controls") / 1000.0
    pitch_ki = cv2.getTrackbarPos("pitch_ki", "Controls") / 100000.0
    pitch_kd = cv2.getTrackbarPos("pitch_kd", "Controls") / 100000.0

    vel_rpm = max(1, cv2.getTrackbarPos("vel_rpm", "Controls"))
    acc = max(1, cv2.getTrackbarPos("acc", "Controls"))

    pid_yaw.set_Kp(yaw_kp)
    pid_yaw.set_Ki(yaw_ki)
    pid_yaw.set_Kd(yaw_kd)

    pid_pitch.set_Kp(pitch_kp)
    pid_pitch.set_Ki(pitch_ki)
    pid_pitch.set_Kd(pitch_kd)

    return vel_rpm, acc


def main():
    print("\n 视觉正常启动\n   [按 'q' / 按 Ctrl+C 退出]")

    LASER_CAM_BASELINE = 0.03  # 激光器到相机光心的水平距离（米）

    # [硬件屏蔽]
    try:
        stepper_yaw.emm_v5_en_control(state=True)
        stepper_pitch.emm_v5_en_control(state=True)
    except Exception as e:  # noqa: F841
        pass

    if SHOW_WINDOWS:
        init_board()

    current_mode = "TRACK"  # 默认工作模式
    prev_time = time.time()

    # 渲染跳帧计数器
    render_counter = 0
    vel_rpm, acc = 500, 100  # 默认参数设定

    try:
        while True:
            render_counter += 1

            # [硬件屏蔽] heart_beat.flash()

            # --- 1. 快速读帧 ---
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # --- 2. 视觉解算 ---
            target = detector.detect(frame)
            yaw_err, pitch_err, dist, status, laser_pos, smooth_center, aim_point = (
                tracker.track(
                    target, mode=current_mode, real_radius_m=0.15, period_sec=3.0
                )
            )
            # --- 2.5 视差补偿（激光在相机左侧 -> 弹道偏右 -> 需要减小 yaw_err）---
            if dist > 0 and status != Status.LOST:
                # 补偿角度（弧度）= arctan(基线 / 距离)
                comp_rad = math.atan2(LASER_CAM_BASELINE, dist)
                comp_deg = math.degrees(comp_rad)  # 转换为度（如果 yaw_err 是度）
                yaw_err -= comp_deg  # 向左补偿
                # 如果你的 yaw_err 是像素单位，需要先转换为角度，或补偿也用像素
                # 具体取决于 tracker 内部实现。通常 tracker.track() 返回的 yaw_err 已经是角度制。

            # --- 3. FPS 计算 ---
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # [硬件屏蔽] 激光射击决策
            # if tracker.onfire: lazer.set_value(1)
            # else: lazer.set_value(0)

            # --- 4. PID 控制解算 ---
            if status in (Status.TRACK, Status.TMP_LOST):
                correction_yaw = pid_yaw.compute(yaw_err)
                correction_pitch = pid_pitch.compute(pitch_err)

                # [硬件屏蔽] 发送给电机
                stepper_yaw.emm_v5_move_to_angle(
                    angle_deg=correction_yaw * GEAR_RATIO_YAW,
                    vel_rpm=vel_rpm,
                    acc=acc,
                    abs_mode=False,
                )
                stepper_pitch.emm_v5_move_to_angle(
                    angle_deg=-correction_pitch * GEAR_RATIO_PITCH,
                    vel_rpm=vel_rpm,
                    acc=acc,
                    abs_mode=False,
                )
            elif status == Status.LOST:
                pid_yaw.reset()
                pid_pitch.reset()

            # --- 5. 防卡顿打印 (每 10 帧打印一次，降低 I/O 阻塞) ---
            if render_counter % 100== 0:
                status_map = {
                    Status.TRACK: "TRACKING",
                    Status.TMP_LOST: "PREDICTING",
                    Status.LOST: "LOST",
                }
                print(
                    f"[{render_counter}] FPS: {fps:>5.1f} | {status_map[status]:<10} | "
                    f"Dist: {dist:>5.2f}m | Yaw_Err: {yaw_err:>6.1f} | Pitch_Err: {pitch_err:>6.1f} | Mode: {current_mode}"
                )

            # ================= 高帧率渲染解耦区 =================
            # 仅在开启界面，且每跑 4 次控制循环才执行 1 次界面渲染 (120Hz -> 30Hz)
            if SHOW_WINDOWS and (render_counter % 4 == 0):
                # 动态读取滑块
                vel_rpm, acc = update_params()

                vis_trk = frame.copy()

                if status != Status.LOST and smooth_center:
                    cv2.drawMarker(
                        vis_trk, smooth_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2
                    )

                if status != Status.LOST and aim_point:
                    cv2.drawMarker(
                        vis_trk, aim_point, (255, 0, 0), cv2.MARKER_CROSS, 15, 2
                    )
                    cv2.line(vis_trk, smooth_center, aim_point, (255, 255, 0), 1)

                if laser_pos:
                    cv2.circle(vis_trk, laser_pos, 4, (0, 0, 255), -1)

                cv2.putText(
                    vis_trk,
                    f"FPS: {fps:.1f} | Mode: {current_mode}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Tracker", vis_trk)

                # 将极其耗时的 waitKey 严格封印在渲染逻辑内
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    current_mode = "CIRCLE" if current_mode == "TRACK" else "TRACK"
                    print(f"\n >>> 模式已切换至: {current_mode} <<< \n")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 收到键盘终止信号...")
    except Exception as e:
        print(f"\n[Error] 主循环发生异常: {str(e)}")
    finally:
        print("\n正在安全释放系统资源...")
        camera.release()

        # [硬件屏蔽]
        try:
            stepper_yaw.emm_v5_stop_now()
            stepper_pitch.emm_v5_stop_now()
            stepper_yaw.emm_v5_en_control(state=False)
            stepper_pitch.emm_v5_en_control(state=False)
            stepper_yaw.close()
            stepper_pitch.close()

        except:  # noqa: E722
            pass
        # lazer.cleanup()
        # heart_beat.cleanup()

        cv2.destroyAllWindows()
        print("系统已安全彻底关闭。")


if __name__ == "__main__":
    main()
