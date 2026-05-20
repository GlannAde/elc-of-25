import math

import cv2

from models.detector import Detector


def calibrate_focal_length():
    # --- 1. 手动设置已知物理参数 ---
    REAL_DIST = 90.0  # 距离 (cm)
    REAL_HEIGHT = 17.5  # 靶纸黑框的真实物理高度 (cm)

    # 这里是 4，如果你的香橙派/电脑读不到画面，记得改成 0 或其他数字
    cap = cv2.VideoCapture(4)

    # 强制设置分辨率，必须与未来比赛运行的分辨率严格一致
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- 2. 初始化检测器 ---
    detector = Detector(min_area=5000, max_area=500000)

    print("--- 标定程序已启动 ---")
    print(f"目标距离设定: {REAL_DIST}cm, 目标物理高度: {REAL_HEIGHT}cm")
    print("操作说明: 按 's' 键确认并保存焦距, 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 1. 运行检测逻辑 (修改为正确的 detect 接口)
        board = detector.detect(frame)

        # 2. 准备绘制画面
        annotated_frame = frame.copy()
        if board is not None:
            # 临时将 board 放进列表以复用 detector 的 draw 方法
            detector.boards = [board]
            annotated_frame = detector.draw(annotated_frame)

        height_px = 0

        # 3. 计算像素高度
        if board is not None and len(board.points) == 4:
            # 计算左侧边和右侧边的平均长度作为像素高度
            pts = board.points
            h_left = math.sqrt(
                (pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2
            )
            h_right = math.sqrt(
                (pts[3][0] - pts[2][0]) ** 2 + (pts[3][1] - pts[2][1]) ** 2
            )
            height_px = (h_left + h_right) / 2.0

            # 4. 实时计算理论焦距 (用于预览)
            current_f = (REAL_DIST * height_px) / REAL_HEIGHT

            # 在画面上显示结果
            cv2.putText(
                annotated_frame,
                f"Current F: {current_f:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Height_px: {height_px:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # 显示检测画面和二值化画面（修改为正确的 self.binary 属性）
        cv2.imshow("Calibration (Annotated)", annotated_frame)
        if detector.binary is not None:
            cv2.imshow("Binary Mask (Check this)", detector.binary)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and height_px > 0:
            final_f = (REAL_DIST * height_px) / REAL_HEIGHT
            print("\n[标定成功!]")
            print(f"检测到像素高度: {height_px:.2f} px")
            print(f"您的像素焦距 f_pixel_h 为: {final_f:.2f}")
            print("-" * 30)
            print(f"请在 Tracker 类中使用这个数字: self.f_pixel_h = {final_f:.2f}")
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate_focal_length()
