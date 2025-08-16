import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import time
import matplotlib
matplotlib.use('Agg')  # Save plots instead of showing
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

        # Parameter to choose trajectory
        self.declare_parameter('use_circular', True)  # Default: circle
        self.use_circular = self.get_parameter('use_circular').value

        # Data storage
        self.start_time = None
        self.times = []
        self.xs = []
        self.ys = []
        self.thetas = []

        # For simulation
        self.sim_x = 5.544444
        self.sim_y = 5.544444
        self.sim_theta = 0.0
        self.sim_times = []
        self.sim_xs = []
        self.sim_ys = []
        self.sim_thetas = []
        self.dt = 0.1

        # Circle parameters
        self.radius = 2.0      # Desired circle radius
        self.v = 1.0           # Linear velocity
        self.w = self.v / self.radius  # Angular velocity = v/r

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
            # also log initial simulated values
            self.sim_times.append(0.0)
            self.sim_xs.append(self.sim_x)
            self.sim_ys.append(self.sim_y)
            self.sim_thetas.append(self.sim_theta)

        current_time = time.time() - self.start_time
        self.times.append(current_time)
        self.xs.append(msg.x)
        self.ys.append(msg.y)
        self.thetas.append(msg.theta)

    def circular_trajectory(self):
        """Generate twist for circular trajectory with radius equation"""
        twist = Twist()
        twist.linear.x = self.v
        twist.angular.z = self.w
        return twist

    def unicycle_kinematics(self, x, y, theta, v, w, dt):
        """Simulated unicycle motion"""
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt
        theta_new = theta + w * dt
        return x_new, y_new, theta_new

    def timer_callback(self):
        if self.start_time is None:
            return

        current_time = time.time() - self.start_time

        # Choose trajectory type (for now only circle used)
        if self.use_circular:
            twist = self.circular_trajectory()
        else:
            twist = Twist()  # placeholder

        # Publish to turtle
        self.publisher_.publish(twist)

        # Simulate trajectory
        self.sim_x, self.sim_y, self.sim_theta = self.unicycle_kinematics(
            self.sim_x, self.sim_y, self.sim_theta,
            twist.linear.x, twist.angular.z, self.dt
        )

        self.sim_times.append(current_time + self.dt)
        self.sim_xs.append(self.sim_x)
        self.sim_ys.append(self.sim_y)
        self.sim_thetas.append(self.sim_theta)

    def plot_trajectories(self):
        if len(self.times) == 0:
            self.get_logger().info("No data to plot.")
            return

        # XY Path
        plt.figure()
        plt.plot(self.xs, self.ys, label='Actual Path')
        plt.plot(self.sim_xs, self.sim_ys, '--', label=f'Desired Circle (r={self.radius})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Circular Trajectory (XY)')
        plt.legend()
        plt.grid(True)
        plt.savefig('circular_xy_with_radius.png')
        plt.close()

        # Theta vs Time
        plt.figure()
        plt.plot(self.times, self.thetas, label='Actual Theta')
        plt.plot(self.sim_times, self.sim_thetas, '--', label='Desired Theta')
        plt.xlabel('Time [s]')
        plt.ylabel('Theta [rad]')
        plt.title('Theta vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('circular_theta_with_radius.png')
        plt.close()

        self.get_logger().info("Plots saved: circular_xy_with_radius.png, circular_theta_with_radius.png")


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
