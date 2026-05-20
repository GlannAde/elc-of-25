#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后台线程只刷新数据；主线程按可配置间隔取“最新一帧”打印。
用法：
  python3 main_bg.py -p /dev/ttyACM0 -b 921600 --print-interval 0.05 --read-sleep 0.001
"""
import argparse
import sys
import time

from models.dm_serial import DM_Serial

def parse_args():
    ap = argparse.ArgumentParser(description="DM IMU reader (background thread refresh, main thread printing)")
    ap.add_argument("-p", "--port", default="/dev/ttyACM1", help="Serial port, e.g. /dev/ttyACM1 or COM3")
    ap.add_argument("-b", "--baud", type=int, default=921600, help="Baudrate, e.g. 921600/460800/115200")
    ap.add_argument("--print-interval", type=float, default=1, help="Seconds between prints in main thread")
    ap.add_argument("--read-sleep", type=float, default=0.001, help="Background reader sleep to limit CPU")
    ap.add_argument("--stats-interval", type=float, default=1.0, help="Seconds between stats prints")
    return ap.parse_args()

def print_stats(dev: DM_Serial):
    total = dev.cnt_ok + dev.cnt_crc + dev.cnt_short + dev.cnt_nohdr
    ok_rate = (dev.cnt_ok / total) if total else 0.0
    print(f"[stats] ok={dev.cnt_ok} crc_fail={dev.cnt_crc} short={dev.cnt_short} nohdr={dev.cnt_nohdr} ok_rate={ok_rate:.3f}", flush=True)

def main():
    args = parse_args()
    dev = DM_Serial(args.port, args.baud)
    if not dev.is_open:
        print(f"[error] cannot open serial: port={args.port} baud={args.baud}\n{dev.last_error()}", file=sys.stderr)
        sys.exit(1)

    if not dev.start_reader(read_sleep=args.read_sleep):
        print("[error] cannot start reader thread", file=sys.stderr)
        sys.exit(1)

    next_print = time.time() + args.print_interval
    next_stats = time.time() + args.stats_interval
    last_count = -1

    try:
        while True:
            now = time.time()

            if now >= next_print:
                pkt, ts, cnt = dev.get_latest()
                if pkt is not None:
                    rid, (f1, f2, f3) = pkt
                    if rid == 0x03:
                        print(f"[{ts:.3f}] RID=0x03  roll={f1:.3f}  pitch={f2:.3f}  yaw={f3:.3f}", flush=True)
                    else:
                        print(f"[{ts:.3f}] RID=0x{rid:02X}  f1={f1:.3f}  f2={f2:.3f}  f3={f3:.3f}", flush=True)
                    last_count = cnt
                else:
                    print("(no data yet)", flush=True)
                next_print = now + args.print_interval

            if now >= next_stats:
                print_stats(dev)
                next_stats = now + args.stats_interval

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        dev.destory()
        print("[info] serial closed.", flush=True)

if __name__ == "__main__":
    main()
