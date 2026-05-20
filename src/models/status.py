import time

import wiringpi

# 全局初始化：使用物理引脚编号 (对应原代码的 GPIO.BOARD)
wiringpi.wiringPiSetupPhys()


class GPIN:
    def __init__(self, pin=1, mode=1):
        self.pin = pin
        self.mode = mode
        self.status = wiringpi.LOW  # 软件电平缓存

        # 主线程心跳检测
        self.last_heartbeat = time.time()
        self.timeout = 3.0  # 3秒超时熔断

        # 呼吸灯状态机
        self._breath_start = 0
        self._pwm_cnt = 0
        self.duty = 0

        # 严格模式隔离配置
        if self.mode == 1:
            wiringpi.pinMode(self.pin, wiringpi.OUTPUT)
            wiringpi.digitalWrite(self.pin, wiringpi.LOW)
        elif self.mode == 0:
            wiringpi.pinMode(self.pin, wiringpi.INPUT)
            # 如果你的按键需要内部上拉电阻，可以取消下面这行的注释：
            # wiringpi.pullUpDnControl(self.pin, wiringpi.PUD_UP)

    def set_value(self, value):
        # 输出模式专属，拦截误写
        if self.mode != 1:
            print(f"错误: 引脚 {self.pin} 是输入模式，无法设置值")
            return

        # 开火极简映射 (保留你原有的反向逻辑：1拉低，0拉高)
        self.status = wiringpi.LOW if value == 1 else wiringpi.HIGH
        wiringpi.digitalWrite(self.pin, self.status)

    def read_status(self):
        # 输入模式专属，拦截误读
        if self.mode == 1:
            print(f"错误: 引脚 {self.pin} 是输出模式，无法读取状态")
            return None

        # 返回1或0, 1 表示按下(低电平)，0 表示松开(高电平)
        return int(wiringpi.digitalRead(self.pin) == wiringpi.LOW)

    def heartbeat(self):
        # 刷新主线程存活时间戳
        self.last_heartbeat = time.time()

    def flash(self):
        # 外部主循环每调用一次，内部推进一帧。绝不 sleep 阻塞业务
        self.heartbeat()
        self._update_breathing()

    def _update_breathing(self):
        # 超时熔断：主线程卡死>3秒，强制拉低灭灯
        if time.time() - self.last_heartbeat > self.timeout:
            wiringpi.digitalWrite(self.pin, wiringpi.LOW)
            self._breath_start = 0  # 重置起点，等待下次恢复
            return

        # 记录呼吸周期起点（首次调用或熔断后重置）
        if self._breath_start == 0:
            self._breath_start = time.time()

        # 固定3秒一个完整周期 (0->100->0)
        elapsed = time.time() - self._breath_start
        phase = (elapsed % 3.0) / 3.0  # 归一化到 0.0 ~ 1.0

        # 三角波计算：前1.5秒占空比线性上升，后1.5秒线性下降
        self.duty = int(phase * 200) if phase < 0.5 else int((1.0 - phase) * 200)

        # 软件PWM输出
        self._pwm_cnt = (self._pwm_cnt + 1) % 100
        current_level = wiringpi.HIGH if self._pwm_cnt < self.duty else wiringpi.LOW
        wiringpi.digitalWrite(self.pin, current_level)

    def cleanup(self):
        # 安全释放，避免残留高电平
        try:
            wiringpi.digitalWrite(self.pin, wiringpi.LOW)
            # WiringPi 没有统一的 cleanup，最安全的做法是把用过的引脚切回输入模式
            wiringpi.pinMode(self.pin, wiringpi.INPUT)
        except Exception:
            pass
