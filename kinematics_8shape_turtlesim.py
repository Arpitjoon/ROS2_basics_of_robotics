import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import signal
import sys


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')

        # Publisher & subscriber
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.subscription = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)

        # Timer at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Data storage
        self.start_time = None
        self.times, self.xs, self.ys, self.thetas = [], [], [], []

        # For simulated trajectory (using kinematics)
        self.sim_x = 5.544444
        self.sim_y = 5.544444
        self.sim_theta = 0.0
        self.sim_times, self.sim_xs, self.sim_ys, self.sim_thetas = [], [], [], []
        self.dt = 0.1

        # Lemniscate parameters
        self.a = 2.0        # scaling
        self.k = 0.5        # speed factor
        self.initial_x = 5.544444
        self.initial_y = 5.544444

        # Offsets for centering
        self.phi0 = -math.asin(math.sqrt(1.0 / 3.0))
        self.lem_x0 = self.a * math.cos(self.phi0) / (math.sin(self.phi0)**2 + 1.0)
        self.lem_y0 = self.a * math.sin(self.phi0) * math.cos(self.phi0) / (math.sin(self.phi0)**2 + 1.0)

        # Register signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.plot_trajectories()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def pose_callback(self, msg):
        if self.start_time is None:
            self.start_time = time.time()
            # log initial simulation
            self.sim_times.append(0.0)
            self.sim_xs.append(self.sim_x)
            self.sim_ys.append(self.sim_y)
            self.sim_thetas.append(self.sim_theta)

        current_time = time.time() - self.start_time
        self.times.append(current_time)
        self.xs.append(msg.x)
        self.ys.append(msg.y)
        self.thetas.append(msg.theta)

    def lemniscate_trajectory(self, t):
        """Compute linear and angular velocity for lemniscate path"""
        phi = self.k * t
        h = 0.001  # finite difference step

        # Positions at phi
        x = self.lem_x(phi)
        y = self.lem_y(phi)

        # Slightly shifted positions for derivative
        xp = self.lem_x(phi + h)
        xm = self.lem_x(phi - h)
        yp = self.lem_y(phi + h)
        ym = self.lem_y(phi - h)

        # First derivatives wrt time
        dxdt = (xp - xm) / (2 * h) * self.k
        dydt = (yp - ym) / (2 * h) * self.k

        # Second derivatives wrt time
        d2xdt2 = (xp - 2*x + xm) / (h**2) * self.k**2
        d2ydt2 = (yp - 2*y + ym) / (h**2) * self.k**2

        v = math.sqrt(dxdt**2 + dydt**2)
        w = 0.0
        if v > 1e-6:
            w = (dxdt * d2ydt2 - dydt * d2xdt2) / (dxdt**2 + dydt**2)

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        return twist

    def lem_x(self, phi):
        psi = phi + self.phi0
        sin_psi = math.sin(psi)
        cos_psi = math.cos(psi)
        den = sin_psi**2 + 1.0
        return self.a * cos_psi / den - self.lem_x0 + self.initial_x

    def lem_y(self, phi):
        psi = phi + self.phi0
        sin_psi = math.sin(psi)
        cos_psi = math.cos(psi)
        den = sin_psi**2 + 1.0
        return self.a * sin_psi * cos_psi / den - self.lem_y0 + self.initial_y

    def unicycle_kinematics(self, x, y, theta, v, w, dt):
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt
        theta_new = theta + w * dt
        return x_new, y_new, theta_new

    def timer_callback(self):
        if self.start_time is None:
            return

        current_time = time.time() - self.start_time

        # Generate velocities for lemniscate
        twist = self.lemniscate_trajectory(current_time)
        self.publisher_.publish(twist)

        # Simulate robot motion with unicycle model
        self.sim_x, self.sim_y, self.sim_theta = self.unicycle_kinematics(
            self.sim_x, self.sim_y, self.sim_theta,
            twist.linear.x, twist.angular.z, self.dt
        )

        # Store simulation values
        self.sim_times.append(current_time + self.dt)
        self.sim_xs.append(self.sim_x)
        self.sim_ys.append(self.sim_y)
        self.sim_thetas.append(self.sim_theta)

    def plot_trajectories(self):
        if len(self.times) == 0:
            self.get_logger().info("No data to plot.")
            return

        # XY trajectory
        plt.figure()
        plt.plot(self.xs, self.ys, label='Actual Path')
        plt.plot(self.sim_xs, self.sim_ys, '--', label='Desired Lemniscate')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Lemniscate Trajectory (XY)')
        plt.legend()
        plt.grid(True)
        plt.savefig('lemniscate_xy.png')
        plt.close()

        # Theta vs time
        plt.figure()
        plt.plot(self.times, self.thetas, label='Actual Theta')
        plt.plot(self.sim_times, self.sim_thetas, '--', label='Desired Theta')
        plt.xlabel('Time [s]')
        plt.ylabel('Theta [rad]')
        plt.title('Theta vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('lemniscate_theta.png')
        plt.close()

        self.get_logger().info("Plots saved: lemniscate_xy.png, lemniscate_theta.png")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.plot_trajectories()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
