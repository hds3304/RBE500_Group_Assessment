#!/usr/bin/env python3
# ik_node.py — Inverse kinematics service for OpenManipulator-like 4-DOF arm

import sys
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from fkik_srv.srv import InvKinService
from sensor_msgs.msg import JointState
from open_manipulator_msgs.srv import SetJointPosition

class InverseKinematicsNode(Node):
    """
    Service server:
        - Service name: /inverse_kinematics
        - Request:  geometry_msgs/Pose pose_values (in mm)
        - Response: sensor_msgs/JointState joint_values
            - name:     ['joint1', 'joint2', 'joint3', 'joint4']
            - position: IK solution in radians
    """

    def __init__(self):
        super().__init__('inverse_kinematics_node')

        # --- Geometry (mm) — keep identical to FK node ---
        self.L0 = self.declare_parameter('L0_mm', 36.076).get_parameter_value().double_value
        self.L1 = self.declare_parameter('L1_mm', 60.25).get_parameter_value().double_value
        self.L2z = self.declare_parameter('L2z_mm', 128.0).get_parameter_value().double_value
        self.L2x = self.declare_parameter('L2x_mm', 24.0).get_parameter_value().double_value
        self.L3 = self.declare_parameter('L3_mm', 124.0).get_parameter_value().double_value
        self.L4 = self.declare_parameter('L4_mm', 133.4).get_parameter_value().double_value

        # Initial guess for IK iterations
        self.current_q = np.zeros(4, dtype=float)

        # Service server
        self.srv = self.create_service(InvKinService, '/get_ik', self.handle_inverse_kinematics)
        self.client = self.create_client(SetJointPosition, '/goal_joint_space_path')
        self.gr_client = self.create_client(SetJointPosition, '/goal_tool_control')

        while not self.client.wait_for_service(timeout_sec=5.0):
            if not rclpy.ok():
                self.get_logger().error('Interrupted while waiting for the service. Exiting.')
                sys.exit(0)
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('InverseKinematicsNode ready. Service: /inverse_kinematics')

    # ---------- Forward Kinematics (same chain as your FK node) ----------

    def quaternion_to_pitch(self, q):
        x, y, z, w = q
        # pitch is rotation about X axis
        sinp = 2 * (w*x + y*z)
        cosp = 1 - 2 * (x*x + y*y)
        return math.atan2(sinp, cosp)

    def remove_base_rotation_from_quaternion(self, q, theta_1):
        """
        Remove the effect of base rotation (theta_1) from input quaternion q.
        """
        # Quaternion for rotation -theta_1 around Z
        cz = math.cos(-theta_1 / 2)
        sz = math.sin(-theta_1 / 2)
        qz_inv = [0.0, 0.0, sz, cz]

        # Original quaternion
        x, y, z, w = q

        # Quaternion multiplication q_rel = qz_inv * q
        xi, yi, zi, wi = qz_inv
        x_rel = wi*x + xi*w + yi*z - zi*y
        y_rel = wi*y - xi*z + yi*w + zi*x
        z_rel = wi*z + xi*y - yi*x + zi*w
        w_rel = wi*w - xi*x - yi*y - zi*z

        return [x_rel, y_rel, z_rel, w_rel]

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        q: array-like of shape (4,) [q1, q2, q3, q4] in radians
        returns: 3D position [x, y, z] in mm (same units as link lengths)
        """
        q1, q2, q3, q4 = q

        T01 = np.array([
            [math.cos(q1), -math.sin(q1), 0.0, 0.0],
            [math.sin(q1),  math.cos(q1), 0.0, 0.0],
            [0.0,           0.0,          1.0, self.L0],
            [0.0,           0.0,          0.0, 1.0]
        ])

        T12 = np.array([
            [math.cos(q2), 0.0,  math.sin(q2), 0.0],
            [0.0,          1.0,  0.0,          0.0],
            [-math.sin(q2), 0.0, math.cos(q2), self.L1],
            [0.0,           0.0, 0.0,          1.0]
        ])

        T23 = np.array([
            [math.cos(q3), 0.0,  math.sin(q3), self.L2x],
            [0.0,          1.0,  0.0,          0.0],
            [-math.sin(q3), 0.0, math.cos(q3), self.L2z],
            [0.0,           0.0, 0.0,          1.0]
        ])

        T34 = np.array([
            [math.cos(q4), 0.0,  math.sin(q4), self.L3],
            [0.0,          1.0,  0.0,          0.0],
            [-math.sin(q4), 0.0, math.cos(q4), 0.0],
            [0.0,           0.0, 0.0,          1.0]
        ])

        # Fixed 90° rotation about y + L4 offset along x
        T4ee = np.array([
            [math.cos(math.pi / 2.0), 0.0,  math.sin(math.pi / 2.0), self.L4],
            [0.0,                     1.0,  0.0,                    0.0],
            [-math.sin(math.pi / 2.0), 0.0, math.cos(math.pi / 2.0), 0.0],
            [0.0,                      0.0, 0.0,                    1.0]
        ])

        T = T01 @ T12 @ T23 @ T34 @ T4ee
        p = T[0:3, 3]  # (x, y, z) in mm
        return p

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Compute 3x4 Jacobian numerically using central differences:
            J_ij = d p_i / d q_j
        """
        J = np.zeros((3, 4), dtype=float)

        for j in range(4):
            dq = np.zeros(4, dtype=float)
            dq[j] = eps

            p_plus = self.forward_kinematics(q + dq)
            p_minus = self.forward_kinematics(q - dq)

            J[:, j] = (p_plus - p_minus) / (2.0 * eps)

        return J

    # ---------- IK Solver (Damped Least Squares) ----------

    def solve_ik(self, target_pos_mm: np.ndarray):
        """
        target_pos_mm: np.array([x, y, z]) in mm
        returns: (q, success, msg)
        """
        q = self.current_q.copy()
        max_iters = 200
        tol = 1e-2  # mm tolerance on position
        damping = 0.01  # DLS damping factor

        for i in range(max_iters):
            p = self.forward_kinematics(q)
            error = target_pos_mm - p
            err_norm = np.linalg.norm(error)

            if err_norm < tol:
                self.get_logger().info(f'IK converged in {i} iterations, error = {err_norm:.4f} mm')
                self.current_q = q
                return q, True, f'Converged with position error {err_norm:.4f} mm'

            J = self.numerical_jacobian(q)

            # Damped least-squares: dq = J^T (J J^T + λ^2 I)^-1 e
            JJt = J @ J.T
            lambda2_I = (damping ** 2) * np.eye(3)
            try:
                inv_term = np.linalg.inv(JJt + lambda2_I)
            except np.linalg.LinAlgError:
                return q, False, 'Jacobian inversion failed (singular)'

            dq = J.T @ (inv_term @ error)

            # Optional: limit step size
            step_norm = np.linalg.norm(dq)
            max_step = 0.1  # rad
            if step_norm > max_step:
                dq *= max_step / step_norm

            q = q + dq

        # If we reach here, IK did not converge
        err_final = np.linalg.norm(self.forward_kinematics(q) - target_pos_mm)
        msg = f'IK did not converge after {max_iters} iterations. Final error: {err_final:.4f} mm'
        self.get_logger().warn(msg)
        return q, False, msg

    # ---------- Service callback ----------

    def handle_inverse_kinematics(self, request, response):
        # Request has: geometry_msgs/Pose pose_values
        pos = request.pose_values.position
        pitch = request.gripper_orientation
        gripper_pos = request.gripper_value

        target_pos_mm = np.array([pos.x, pos.y, pos.z], dtype=float)

        self.get_logger().info(
            f'Received IK request for target position (mm): '
            f'[{target_pos_mm[0]:.2f}, {target_pos_mm[1]:.2f}, {target_pos_mm[2]:.2f}]'
        )

        q, success, msg = self.solve_ik(target_pos_mm)

        # Fill a JointState for the response
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = 'base_link'  # or whatever you use
        js.name = ['joint1', 'joint2', 'joint3', 'joint4']
        js.position = [float(a) for a in q.tolist()]
        js.velocity = []  # can leave empty
        js.effort = []    # can leave empty

        response.joint_values = js

        if success:
            q_vals = [float(a) for a in q.tolist()]
            q_vals[3] = math.radians(pitch)

            self.get_logger().info(
                f'IK success, joints (rad)=\n'
                f'[{q_vals[0]:.3f}, {q_vals[1]:.3f}, {q_vals[2]:.3f}, {q_vals[3]:.3f}]\n'
            )
            self.send_request((*q_vals, 0.5))
            time.sleep(2.5)
            self.send_gripper_request(gripper_pos)
            time.sleep(2.5)
        else:
            self.get_logger().warn(f'IK failed: {msg}')

        return response

    def send_request(self, positions):
        # Create a request object
        request = SetJointPosition.Request()

        # Set the request parameters
        request.planning_group = 'arm'
        request.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        request.joint_position.position = positions
        request.path_time = 2.0

        # Send the request asynchronously and wait for the result
        self.client.call_async(request)
        self.get_logger().info("Successfully called goal_joint_client with joint angles.")

    def send_gripper_request(self, position):
        req = SetJointPosition.Request()
        req.planning_group = "gripper"
        req.joint_position.joint_name = ["gripper"]
        req.joint_position.position = [position]
        req.path_time = 2.0

        self.gr_client.call_async(req)
        self.get_logger().info("Successfully called goal_tool_control with joint angles")


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematicsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
