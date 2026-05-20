import time

import cv2

# 只保留视觉相关的模块
from models.cam import Camera
from models.detector import Detector


def main():
    print("启动纯视觉测试模式")

    # --- 1. 初始化相机 ---
    # 记得核对 index，如果是外部 USB 摄像头通常是 0, 1, 4 等
    cam = Camera(index=0, width=640, height=480, format="MJPG", fps=120)
    center_x, center_y = cam.width / 2, cam.height / 2

    # --- 2. 初始化视觉大脑 ---
    # 如果环境光线复杂导致 FPS 变低，用 adaptive threshold，并在 detector 中手动定阈值
    detector = Detector(min_area=5000, max_area=500000, use_adaptive=True)

    prev_time = time.time()

    try:
        while True:
            # A. 极速读取最新帧
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            # B. 耗时与 FPS 计算
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / (dt + 1e-6)
            prev_time = curr_time

            # C. 视觉识别 (核心算力层)
            # 如果你想看每一步的耗时，可以把 debug 设为 True
            target_board = detector.detect(frame, debug=False)

            # D. 状态判断与可视化
            if target_board and target_board.center:
                target_x, target_y = target_board.center
                # 识别到目标，在目标中心画一个绿色准星
                cv2.drawMarker(
                    frame,
                    (int(target_x), int(target_y)),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )
                state_str = "DETECTED"
                color = (0, 255, 0)
            else:
                state_str = "LOST"
                color = (0, 0, 255)

            # 画出画面正中心的基准十字 (红色)
            cv2.drawMarker(
                frame,
                (int(center_x), int(center_y)),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                20,
                1,
            )

            # 屏幕左上角打印 FPS 和状态
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | {state_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # 叠加 detector 内部的绘制画面 (靶纸边框等)
            frame = detector.draw(frame)
            cv2.imshow("Vision Test Mode", frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n收到退出指令...")
    except Exception as e:
        print(f"\n主循环异常: {e}")
    finally:
        print("正在安全释放相机资源...")
        cam.release()
        cv2.destroyAllWindows()
        print("视觉测试已结束。")


if __name__ == "__main__":
    main()
