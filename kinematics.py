import rclpy
from rclpy.node import Node
import math

class KinematicsNode(Node):
    def __init__(self):
        super().__init__('kinematics_node')
        # Initial pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.dt = 0.1  # time step

    def unicycle_kinematics(self, x, y, theta, v, w, dt):
        """
        Update the robot pose using unicycle model
        """
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt
        theta_new = theta + w * dt
        return x_new, y_new, theta_new

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()

    # Run the kinematics update loop for N steps
    steps = 5
    for i in range(steps):
        try:
            v = float(input(f"Step {i+1}/{steps} - Enter linear velocity v: "))
            w = float(input(f"Step {i+1}/{steps} - Enter angular velocity w: "))
        except ValueError:
            print("Invalid input, please enter numbers.")
            continue

        # Update pose
        node.x, node.y, node.theta = node.unicycle_kinematics(
            node.x, node.y, node.theta, v, w, node.dt
        )

        # Print result
        print(f"Pose after step {i+1}: x={node.x:.3f}, y={node.y:.3f}, theta={node.theta:.3f}")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
