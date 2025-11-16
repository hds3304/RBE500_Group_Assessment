#!/usr/bin/env python3
import time
import math
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from open_manipulator_msgs.srv import SetJointPosition


class ForwardKinematicsNode(Node):
    """
    Goal:
        1. Recieve Joint Position command from user.
        2. Move the robot to user's given position.
        3. Calculate the current position using /joint_states
    """
    def __init__(self):
        super().__init__('forward_kinematics_node')
        # self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.publisher_ = self.create_subscription(JointState, '/goal_joiint_state', self.joint_cb, 10)

        # --- Geometry (mm) — override via ROS params if needed ---
        self.L0 = self.declare_parameter('L0_mm', 36.076).get_parameter_value().double_value
        self.L1 = self.declare_parameter('L1_mm', 60.25).get_parameter_value().double_value
        self.L2z = self.declare_parameter('L2z_mm', 128.0).get_parameter_value().double_value
        self.L2x = self.declare_parameter('L2x_mm', 24.0).get_parameter_value().double_value
        self.L3 = self.declare_parameter('L3_mm', 124.0).get_parameter_value().double_value
        self.L4 = self.declare_parameter('L4_mm', 133.4).get_parameter_value().double_value
        
        self.last_joints = None
        self.cli = self.create_client(SetJointPosition, '/goal_joint_space_path')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /goal_joint_space_path service...')
        self.last_joints = [0.0, 0.0, 0.0, 0.0]
        self.send_to_robot(self.last_joints)

    def rotmat_to_quat(self, R: np.ndarray):
        """Convert a 3x3 rotation matrix to (x, y, z, w) quaternion."""
        t = np.trace(R)
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            # find the largest diagonal element
            i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
            if i == 0:
                s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
                w = (R[2, 1] - R[1, 2]) / s
            elif i == 1:
                s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
                w = (R[0, 2] - R[2, 0]) / s
            else:
                s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
                w = (R[1, 0] - R[0, 1]) / s
        return float(x), float(y), float(z), float(w)


    def deg2rad(self, theta):
        return math.radians(theta)

    def send_to_robot(self, joint_rad):
        """Moves the robot to given joints (in radians)"""
        if not self.cli.service_is_ready():
            self.get_logger().error("Service not available — controller may have crashed.")
            return

        req = SetJointPosition.Request()
        req.planning_group = ""
        req.joint_position.joint_name = ['joint1','joint2','joint3','joint4','gripper']
        req.joint_position.position = joint_rad + [0.0]  # Adding gripper joint value
        req.path_time = 3.0
        self.cli.call_async(req)
        time.sleep(5)  # Sleep timer to let the robot command finish cleanly

        self.get_logger().info(f"Sent robot to (radians): {joint_rad}")

    def joint_cb(self, msg: JointState):
        q = list(msg.position[:4])
        self.get_logger().info(f"Commanding robot to move to (degrees): {q}")

        # Convert joint values from radians to degrees
        self.last_joints = q
        rq1, rq2, rq3, rq4 = [self.deg2rad(a) for a in q] 
        self.send_to_robot([rq1, rq2, rq3, rq4])  # Expects radians

        T01 = np.array([
            [math.cos(rq1), -math.sin(rq1), 0, 0],
            [math.sin(rq1), math.cos(rq1), 0, 0],
            [0, 0, 1, self.L0],
            [0, 0, 0, 1]])
        T12 = np.array([
            [math.cos(rq2), 0, math.sin(rq2), 0],
            [0, 1, 0, 0],
            [-math.sin(rq2), 0, math.cos(rq2), self.L1],
            [0, 0, 0, 1]])
        T23 = np.array([
            [math.cos(rq3), 0, math.sin(rq3), self.L2x],
            [0, 1, 0, 0],
            [-math.sin(rq3), 0, math.cos(rq3), self.L2z],
            [0, 0, 0, 1]])
        T34 = np.array([
            [math.cos(rq4), 0, math.sin(rq4), self.L3],
            [0, 1, 0, 0],
            [-math.sin(rq4), 0, math.cos(rq4), 0],
            [0, 0, 0, 1]])
        T4ee = np.array([
            [math.cos((np.pi/2)), 0, math.sin((np.pi/2)), self.L4],
            [0, 1, 0, 0],
            [-math.sin((np.pi/2)), 0, math.cos((np.pi/2)), 0],
            [0, 0, 0, 1]])

        T = T01@T12@T23@T34@T4ee

        R = T[:3, :3]  # I need it for IK do not remove
        p = T[:3, 3]

        # Quaternions
        qx, qy, qz, qw = self.rotmat_to_quat(R)

        self.get_logger().info(f"EE Position (x, y, z): [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        self.get_logger().info(f"EE Orientation (x, y, z, w): [{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}]")


def main():
    rclpy.init()
    node = ForwardKinematicsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

## -ve angles mean that the motor is moving clockwise
## ros2 topic pub --once /joint_states sensor_msgs/JointState "{name: ['joint1','joint2','joint3','joint4','gripper'],position: [0.0, 0.0, 0.0, 0.0, 0.0]}"> /dev/null