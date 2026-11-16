#!/usr/bin/env python3
"""RoboCar identification runs (Section 6.3); one run is written to one CSV file.

ROS 2, run on the vehicle. Commands are AckermannDriveStamped messages on drive_ad (input
selected through mux_drive/select); measured speed from drive_status, IMU from
imu/data_raw, battery voltage from motor_battery. Steering is held at 0 (straight line).

The vehicle takes a speed command, so the acceleration input u is applied as
    --cmd-mode ramp      v_cmd(k+1) = v_cmd(k) + u(k) dt        (default)
    --cmd-mode inverse   v_cmd(k)   = v_meas(k) + tau_inv u(k)
    --cmd-mode feedback  v_cmd(k+1) = v_meas(k) + u(k) dt       (not used in the paper)

Tests (--test):
  speed     cruise at --v for --dist metres (integrated speed), then stop
  accel     from rest, u = +u_cmd until --v, hold 1 s, stop
  brake     ramp up to --v, hold 1 s, then u = -u_cmd until stopped
  reversal  ramp up, hold, u = +u_cmd for --t-rev s, then brake (not used in the paper)
  random    seeded piecewise-constant u in [-u_brake, +u_accel], held 0.3-1.0 s,
            speed kept in [0, --v]; duration --dur s

Safety: speed cap --vmax, distance stop --track, Ctrl-C stops the vehicle.

Example:
  python3 robocar_ident.py --test brake --cmd-mode inverse --tau-inv 0.4 --u-cmd 2.5 \
      --v 0.6 --track 3.5 --run-id BRKI-06-2 --ros-args -r __ns:=/rc_23005

The ramp-mode runs were recorded before the inverse mode was added; their header line has
no tau_inv field. The code paths used by those runs are unchanged.
"""
import argparse, csv, math, os, random, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, BatteryState
from ackermann_msgs.msg import AckermannDriveStamped

try:
    from rc110_topic_tools.srv import MuxSelect
except ImportError:
    MuxSelect = None


class Ident(Node):
    def __init__(self, a):
        super().__init__("robocar_ident")
        self.a = a
        self.dt = 1.0 / a.rate
        self.pub = self.create_publisher(AckermannDriveStamped, "drive_ad", 10)
        self.create_subscription(AckermannDriveStamped, "drive_status", self.on_status, 50)
        self.create_subscription(Imu, "imu/data_raw", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(BatteryState, "motor_battery", self.on_batt, 10)
        self.v_meas = 0.0; self.t_status = None; self.n_status = 0
        self.ax = float("nan"); self.t_imu = None
        self.batt_v = float("nan")
        self.v_cmd = 0.0; self.x_int = 0.0
        self.phase = "start"; self.phase_t0 = None; self.t0 = None
        self.rng = random.Random(a.seed)
        self.u_rand = 0.0; self.rand_until = 0.0
        self.done = False
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"ident_{a.run_id}.csv")
        self.f = open(fn, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["# run_id", a.run_id, "test", a.test, "cmd_mode", a.cmd_mode,
                         "rate", a.rate, "tau_inv", a.tau_inv, "v", a.v, "u_cmd", a.u_cmd, "seed", a.seed,
                         "floor", a.floor, "start_unix", f"{time.time():.3f}"])
        self.w.writerow(["t", "t_pub_wall", "phase", "u_cmd", "v_cmd", "v_meas",
                         "status_age", "n_status", "x_int", "imu_ax", "imu_age", "batt_v"])
        self.get_logger().info(f"logging to {fn}")
        self.select("drive_ad")
        self.timer = self.create_timer(self.dt, self.step)

    def select(self, topic):
        if MuxSelect is None:
            self.get_logger().warn("MuxSelect not available: select drive_ad manually"); return
        cli = self.create_client(MuxSelect, "mux_drive/select")
        if not cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("mux_drive/select not available"); return
        req = MuxSelect.Request(); req.topic = topic
        cli.call_async(req)

    def on_status(self, m):
        self.v_meas = m.drive.speed; self.t_status = time.monotonic(); self.n_status += 1

    def on_imu(self, m):
        self.ax = m.linear_acceleration.x; self.t_imu = time.monotonic()

    def on_batt(self, m):
        self.batt_v = m.voltage

    def send(self, v):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.drive.speed = float(v); m.drive.steering_angle = 0.0
        self.pub.publish(m)

    def set_phase(self, p, now):
        self.phase = p; self.phase_t0 = now
        self.get_logger().info(f"phase -> {p}")

    def u_law(self, now):
        a = self.a; tp = now - self.phase_t0
        if a.test == "speed":
            if self.phase == "start": self.set_phase("ramp", now); return a.u_cmd
            if self.phase == "ramp":
                if self.v_cmd >= a.v: self.set_phase("cruise", now); return 0.0
                return a.u_cmd
            if self.phase == "cruise":
                if self.x_int >= a.dist: self.set_phase("stop", now); return -a.u_cmd
                return 0.0
        elif a.test == "accel":
            if self.phase == "start": self.set_phase("accel", now); return a.u_cmd
            if self.phase == "accel":
                if self.v_cmd >= a.v: self.set_phase("hold", now); return 0.0
                return a.u_cmd
            if self.phase == "hold":
                if tp >= 1.0: self.set_phase("stop", now); return -0.5
                return 0.0
        elif a.test in ("brake", "reversal"):
            if self.phase == "start": self.set_phase("ramp", now); return a.u_ramp
            if self.phase == "ramp":
                if self.v_cmd >= a.v: self.set_phase("hold", now); return 0.0
                return a.u_ramp
            if self.phase == "hold":
                if tp >= 1.0:
                    if a.test == "brake": self.set_phase("brake", now); return -a.u_cmd
                    self.set_phase("plus", now); return a.u_cmd
                return 0.0
            if self.phase == "plus":
                if tp >= a.t_rev: self.set_phase("brake", now); return -a.u_cmd
                return a.u_cmd
            if self.phase == "brake":
                return -a.u_cmd
        elif a.test == "random":
            if self.phase == "start": self.set_phase("random", now)
            if self.phase == "random":
                if now - self.t0 >= a.dur: self.set_phase("stop", now); return -a.u_brake
                if now >= self.rand_until:
                    self.u_rand = self.rng.uniform(-a.u_brake, a.u_accel)
                    self.rand_until = now + self.rng.uniform(0.3, 1.0)
                u = self.u_rand
                if self.v_cmd <= 0.05 and u < 0: u = abs(u)          # stay moving
                if self.v_cmd >= a.v and u > 0: u = -abs(u)          # stay under v
                return u
        if self.phase == "stop":
            return -min(a.u_cmd, 1.0)                         # controlled stop
        return -a.u_cmd if self.phase == "brake" else -0.5

    def step(self):
        if self.done: return
        now = time.monotonic()
        if self.t0 is None: self.t0 = now; self.phase_t0 = now
        a = self.a
        u = self.u_law(now)
        if a.cmd_mode == "ramp":
            v_next = self.v_cmd + u * self.dt
        elif a.cmd_mode == "inverse":                     # v_cmd = v + tau_inv * u
            v_next = self.v_meas + u * a.tau_inv
        else:
            v_next = self.v_meas + u * self.dt
        v_next = max(0.0, min(a.vmax, v_next))
        self.x_int += self.v_meas * self.dt
        if self.x_int >= a.track:                       # distance safety stop
            v_next = 0.0
            if self.phase != "track_stop": self.set_phase("track_stop", now)
        self.v_cmd = v_next
        self.send(self.v_cmd)
        st_age = (now - self.t_status) if self.t_status else float("nan")
        imu_age = (now - self.t_imu) if self.t_imu else float("nan")
        self.w.writerow([f"{now - self.t0:.4f}", f"{time.time():.4f}", self.phase, f"{u:.4f}",
                         f"{self.v_cmd:.4f}", f"{self.v_meas:.4f}", f"{st_age:.4f}", self.n_status,
                         f"{self.x_int:.4f}", f"{self.ax:.4f}", f"{imu_age:.4f}", f"{self.batt_v:.3f}"])
        stopped = self.v_cmd == 0.0 and abs(self.v_meas) < 0.02 and now - self.t0 > 1.0
        if (self.phase in ("stop", "brake", "track_stop") and stopped) or now - self.t0 > a.timeout:
            self.finish()

    def finish(self):
        for _ in range(5): self.send(0.0)
        self.done = True; self.f.close()
        self.get_logger().info(f"run {self.a.run_id} done, integrated distance {self.x_int:.3f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=["speed", "accel", "brake", "reversal", "random"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--v", type=float, default=0.6, help="target / cap speed [m/s]")
    ap.add_argument("--u-cmd", type=float, default=2.0, help="step acceleration magnitude [m/s^2]")
    ap.add_argument("--u-ramp", type=float, default=0.8, help="gentle ramp-up before brake/reversal [m/s^2]")
    ap.add_argument("--u-accel", type=float, default=1.5, help="random: max accel [m/s^2]")
    ap.add_argument("--u-brake", type=float, default=2.0, help="random: max braking [m/s^2]")
    ap.add_argument("--t-rev", type=float, default=0.5, help="reversal: +u duration [s]")
    ap.add_argument("--dist", type=float, default=3.0, help="speed test: cruise distance [m]")
    ap.add_argument("--dur", type=float, default=8.0, help="random test duration [s]")
    ap.add_argument("--cmd-mode", default="ramp", choices=["ramp", "feedback", "inverse"])
    ap.add_argument("--tau-inv", type=float, default=0.25, help="inverse mode: v_cmd = v + tau_inv * u [s]")
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--track", type=float, default=5.5, help="hard distance stop [m]")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", default="tile")
    ap.add_argument("--out", default="ident_logs")
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    n = Ident(a)
    try:
        while rclpy.ok() and not n.done:
            rclpy.spin_once(n, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if not n.done:
            for _ in range(5): n.send(0.0)
            n.f.close()
        n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
