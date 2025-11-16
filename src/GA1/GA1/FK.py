#!/usr/bin/env python3
import sys
import math
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose


class ForwardKinematicsNode(Node):
    def __init__(self):
        super().__init__('forward_kinematics_node')
        self.subscription = self.create_subscription(JointState, '/joint_states',self.joint_cb, 10)
        self.publisher_ = self.create_publisher(Pose, '/end_effector_pose', 10)

        # --- Geometry (mm) — override via ROS params if needed ---
        self.L0 = self.declare_parameter('L0_mm', 36.076).get_parameter_value().double_value
        self.L1 = self.declare_parameter('L1_mm', 60.25).get_parameter_value().double_value
        self.L2z = self.declare_parameter('L2z_mm', 128.0).get_parameter_value().double_value
        self.L2x = self.declare_parameter('L2x_mm', 24.0).get_parameter_value().double_value
        self.L3 = self.declare_parameter('L3_mm', 124.0).get_parameter_value().double_value
        self.L4 = self.declare_parameter('L4_mm', 133.4).get_parameter_value().double_value
        self.last_pose = [math.inf]*4

    def deg2rad(self, theta):
        return math.radians(theta)

    def joint_cb(self, msg: JointState):

        q1, q2, q3, q4 = msg.position[:4]
        self.last_pose = [q1, q2, q3, q4]

        rq1 = self.deg2rad(q1)
        rq2 = self.deg2rad(q2)
        rq3 = self.deg2rad(q3)
        rq4 = self.deg2rad(q4)

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

        R = T[:3, :3]
        p = T[:3, 3]

        self.get_logger().info(f"EE Position: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")


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
## ros2 topic pub --once /joint_states sensor_msgs/JointState "{name: ['joint1','joint2','joint3','joint4'],position: [0.0, 0.0, 0.0, 0.0]}"> /dev/null

