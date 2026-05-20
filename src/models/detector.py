import time

import cv2
import numpy as np


class Board:
    def __init__(self):
        self.points = []  # 四个端点坐标
        self.center = None  # 对角线交点坐标 (x, y)
        self.area = 0.0


class Detector:
    # 注意：这里的 use_otsu 改为了 use_adaptive 更加名副其实
    def __init__(self, min_area=3000, max_area=500000, use_adaptive=True):
        self.board_min_area = min_area
        self.board_max_area = max_area
        self.use_adaptive = use_adaptive
        self.manual_threshold = 127  # 如果关闭自适应，则使用此手动阈值
        self.boards = []
        self.raw = None
        self.binary = None

        # 预留标准的正方形坐标，用于透视变换
        self.std_square = np.float32([[0, 0], [0, 100], [100, 100], [100, 0]])

        # --- [新增] 记忆上一帧的靶纸中心位置 ---
        self.last_center = None

    def process_image(self, frame):
        self.raw = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- [核心优化 1]：放弃全局 OTSU，改用局部自适应阈值 ---
        # 这样能无视赛场阴影、反光等不均匀光照，完美抠出黑边框
        if self.use_adaptive:
            # 参数说明：
            # 11: 局部计算窗口大小，必须是奇数 (如 11, 15, 21)。太大会变回全局，太小抗噪差
            # 2:  常数补偿值。如果发现框断断续续，可以调小(比如0)；如果发现噪点多，可以调大(比如5)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binary = cv2.threshold(
                gray, self.manual_threshold, 255, cv2.THRESH_BINARY_INV
            )

        self.binary = binary
        return binary

    def find_board(self, binary):
        """保留强大的 RETR_CCOMP 拓扑查找逻辑，增加几何防线"""
        boards = []
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            self.boards = []
            return []

        # 优先寻内轮廓，没有则寻外轮廓
        inner_contours = [
            (i, c) for i, c in enumerate(contours) if hierarchy[0][i][3] != -1
        ]
        target_contours = (
            inner_contours
            if inner_contours
            else [(i, c) for i, c in enumerate(contours) if hierarchy[0][i][3] == -1]
        )

        for i, contour in target_contours:
            area = cv2.contourArea(contour)
            if self.board_min_area < area < self.board_max_area:
                peri = cv2.arcLength(contour, True)
                # 拟合多边形，容差为周长的 2%
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # 必须是四边形
                if len(approx) == 4:
                    # --- [核心优化 2.1]：凸包性校验 ---
                    # 物理世界的靶纸透视畸变后必定是凸多边形，排除凹陷的干扰物
                    if not cv2.isContourConvex(approx):
                        continue

                    # --- [核心优化 2.2]：长宽比过滤 ---
                    # 排除电线、细长支架、门缝等长条形的黑色干扰
                    x, y, w, h = cv2.boundingRect(approx)
                    # max(h, 1) 是防止极端情况下高度为 0 导致除零崩溃
                    aspect_ratio = float(w) / max(h, 1)
                    # 靶纸是A4，这里放宽比例到 0.8 ~ 1.8
                    if not (0.8 < aspect_ratio < 1.8):
                        continue

                    # 1. 立即转为纯 Python 列表，抛弃 Numpy 包袱
                    pts = approx.reshape(4, 2).tolist()

                    # 2. 根据 x+y 和 x-y 的特征快速定位四个角
                    tl = min(pts, key=lambda p: p[0] + p[1])  # 左上
                    br = max(pts, key=lambda p: p[0] + p[1])  # 右下
                    bl = max(pts, key=lambda p: p[0] - p[1])  # 左下
                    tr = min(pts, key=lambda p: p[0] - p[1])  # 右上

                    sorted_points = [tl, bl, br, tr]

                    # 3. 容错校验（防止畸变导致点重合）
                    if len(set(tuple(pt) for pt in sorted_points)) < 4:
                        pts_x = sorted(pts, key=lambda p: p[0])
                        pts_y = sorted(pts, key=lambda p: p[1])
                        sorted_points = [
                            pts_x[0],  # 最左
                            pts_y[0],  # 最上
                            pts_x[-1],  # 最右
                            pts_y[-1],  # 最下
                        ]

                    # 4. 构建 Board 对象
                    board = Board()
                    board.points = [(int(pt[0]), int(pt[1])) for pt in sorted_points]
                    board.area = area
                    board.center = self._calculate_intersection(board.points)

                    if board.center is not None:
                        boards.append(board)

        # 按面积从大到小排序，取最大的目标
        if boards:
            boards.sort(key=lambda b: b.area, reverse=True)
            self.boards = boards
        else:
            self.boards = []

        return boards

    def get_perspective_offset(self, board, target_ratio_x=0.5, target_ratio_y=0.5):
        """透视变换：精准定位到靶纸内部的任意比例点"""
        if len(board.points) != 4:
            return board.center

        dst_pts = np.float32(board.points)
        M = cv2.getPerspectiveTransform(self.std_square, dst_pts)

        # 目标点在 100x100 标准正方形里的坐标
        target_pt = np.float32([[[100 * target_ratio_x, 100 * target_ratio_y]]])

        # 映射回现实畸变的画面中
        real_pt = cv2.perspectiveTransform(target_pt, M)
        return (int(real_pt[0][0][0]), int(real_pt[0][0][1]))

    def _calculate_intersection(self, points):
        """两点式求交点 (保留原版)"""
        x1, y1 = points[0]
        x2, y2 = points[2]
        x3, y3 = points[1]
        x4, y4 = points[3]
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None
        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator
        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator
        return (int(px), int(py))

    def draw(self, image):
        if not self.boards or image is None:
            return image

        board = self.boards[0]
        if not board.points or board.center is None:
            return image

        pts = np.array(board.points, np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        # 蓝色画出对角线
        cv2.line(image, board.points[0], board.points[2], (255, 0, 0), 2)
        cv2.line(image, board.points[1], board.points[3], (255, 0, 0), 2)

        # 绿色画出交点
        cv2.circle(image, board.center, 5, (0, 255, 0), -1)

        # 画出相机光轴中心
        h, w = image.shape[:2]
        cv2.circle(image, (w // 2, h // 2), 5, (0, 165, 255), -1)

        return image

    def detect(self, frame, debug=False):
        start = time.time()
        self.raw = frame
        h, w = frame.shape[:2]

        boards = []

        # ================= ROI 局部极速搜索 =================
        if self.last_center is not None:
            cx, cy = self.last_center
            roi_half = 120  # 搜索半径 (可以根据云台运动速度微调)

            # 计算 ROI 边界，防止数组越界
            x1 = max(0, int(cx - roi_half))
            y1 = max(0, int(cy - roi_half))
            x2 = min(w, int(cx + roi_half))
            y2 = min(h, int(cy + roi_half))

            # --- [新增防线]：如果抠出来的图太小甚至长宽为0，直接放弃局部搜索，转入全局 ---
            if x2 - x1 < 30 or y2 - y1 < 30:
                self.last_center = None
            else:
                # 抠出局部小图
                roi_frame = frame[y1:y2, x1:x2]

                # --- 遗漏的纯色背景防爆炸优化 ---
                gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.meanStdDev(gray_roi)[1][0][0] ** 2

                if variance < 50.0:
                    self.last_center = None
                else:
                    # 只对这块小图做高耗时的自适应阈值
                    bin_roi = self.process_image(roi_frame)
                    boards = self.find_board(bin_roi)

                # 把局部坐标换算回全局坐标 (非常关键！)
                for board in boards:
                    board.points = [(px + x1, py + y1) for px, py in board.points]
                    board.center = (board.center[0] + x1, board.center[1] + y1)

                # 如果局部搜索成功，更新中心点并直接返回，跳过全局搜索！
                if boards:
                    self.last_center = boards[0].center
                    self.boards = boards
                    if debug:
                        print(
                            f"Vision Cost (ROI): {(time.time() - start) * 1000:.1f}ms"
                        )
                    return boards[0]
                else:
                    self.last_center = None

        # ================= 全局搜索 (降级) =================
        # 只有系统刚启动，或者目标飞出 ROI 框时才会执行这里
        # [核心优化]：全图搜索时缩小画面，将算力消耗降低 75%，防止产生 Low 帧！
        small_frame = cv2.resize(frame, (w // 2, h // 2))

        # 因为画面长宽缩小了一半，面积阈值必须除以 4
        orig_min_area = self.board_min_area
        self.board_min_area = orig_min_area / 4.0

        bin_img = self.process_image(small_frame)
        boards = self.find_board(bin_img)

        # 恢复面积阈值
        self.board_min_area = orig_min_area

        if boards:
            # [极其关键]：将找到的小图坐标，按比例放大回原图的真实坐标
            for b in boards:
                b.points = [(px * 2, py * 2) for px, py in b.points]
                b.center = (b.center[0] * 2, b.center[1] * 2)
                b.area = b.area * 4

            self.last_center = boards[0].center
            self.boards = boards
        else:
            self.last_center = None
            self.boards = []

        if debug:
            print(f"Vision Cost (Global): {(time.time() - start) * 1000:.1f}ms")

        return boards[0] if boards else None

    def display(self, dis):
        if self.raw is None:
            return None, self.binary
        vis = self.raw.copy()
        if dis == 1:
            res = self.draw(vis)
            return res, self.binary
        return vis, self.binary
