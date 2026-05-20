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
        輸入: error (當前誤差)
        返回: output (控制量)
        """
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0 or dt > 1.0:
            dt = 1 / 120.0

        # --- [優化2] 死區控制：切入死區時立即截斷，防止 D 項突變震動 ---
        if abs(error) < self.deadzone:
            self.integral = 0.0
            self.last_error = 0.0
            self.last_derivative = 0.0
            self.last_time = current_time
            return 0.0  # 直接回傳 0，不輸出任何微小抖動指令

        # --- P (比例) ---
        p_out = self.Kp * error

        # --- I (積分) ---
        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_out = self.Ki * self.integral

        # --- D (微分) ---
        if dt > 0:
            raw_derivative = (error - self.last_error) / dt
            alpha = 0.3
            derivative = alpha * raw_derivative + (1.0 - alpha) * self.last_derivative
        else:
            derivative = 0.0

        d_out = self.Kd * derivative

        # --- 更新狀態 ---
        self.last_error = error
        self.last_time = current_time
        self.last_derivative = derivative

        # --- 總輸出與限幅 ---
        output = p_out + i_out + d_out
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
