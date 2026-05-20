# pid模块
import time


class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=1 / 120.0, max_output=None, deadzone=0.2):
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益
        self.dt = dt  # 时间步长

        self.last_error = 0.0  # 上一次的误差,用于计算D
        self.integral = 0.0  # 积分累积值,用于计算I
        self.last_time = time.time()  # 上一次计算的时间戳

        # [优化1] 微分低通滤波历史值
        self.last_derivative = 0.0

        # [优化2] 死区(Deadzone)，例如 0.2 度以内认为已瞄准，不乱动
        self.deadzone = deadzone

        # [优化3] 输出限幅，防止电机接到极其离谱的突变指令
        self.max_output = max_output

        # 抗积分饱和（防止I项过大导致失控）
        self.integral_limit = 100.0

    def compute(self, error):
        """
        输入: error (当前误差，例如 yaw 角度偏差)
        返回: output (控制量，例如电机需要转动的角度增量)
        """
        current_time = time.time()  # 这一次的时间
        dt = current_time - self.last_time

        # 防止 dt 过小或为负数
        if dt <= 0 or dt > 1.0:
            dt = 1 / 120.0  # 默认为高频时间步长

        # --- [优化2] 死区控制 ---
        if abs(error) < self.deadzone:
            error = 0.0
            self.integral = 0.0  # 到了死区就清空积分，防止长时间静止后积分积累导致抽搐

        # --- P (比例) ---
        p_out = self.Kp * error

        # --- I (积分) ---
        self.integral += error * dt
        # 限制积分项的大小，防止失控
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_out = self.Ki * self.integral

        # --- D (微分) ---
        if dt > 0:
            raw_derivative = (error - self.last_error) / dt
            # 优化1 一阶低通滤波 (不完全微分)
            # 公式: D_current = alpha * D_raw + (1 - alpha) * D_last
            # alpha 越小，越平滑，抗噪越好，但响应会稍微变慢。0.3是个不错的经验值。
            alpha = 0.3
            derivative = alpha * raw_derivative + (1.0 - alpha) * self.last_derivative
        else:
            derivative = 0.0

        d_out = self.Kd * derivative

        # --- 更新状态 ---
        self.last_error = error
        self.last_time = current_time
        self.last_derivative = derivative

        # --- 总输出 ---
        output = p_out + i_out + d_out

        # --- 优化3 输出限幅 ---
        if self.max_output is not None:
            if output > self.max_output:
                output = self.max_output
            elif output < -self.max_output:
                output = -self.max_output

        return output

    def reset(self):
        """重置状态"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.last_time = time.time()

    def set_Kp(self, Kp):
        self.Kp = Kp

    def set_Ki(self, Ki):
        self.Ki = Ki

    def set_Kd(self, Kd):
        self.Kd = Kd
