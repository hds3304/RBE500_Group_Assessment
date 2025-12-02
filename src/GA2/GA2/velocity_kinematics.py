#!/usr/bin/env python3
import rclpy
import numpy as np
import math
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Import your services
from ga2_srv.srv import ForwardVelocity, InverseVelocity

class VelocityKinematicsNode(Node):
    def __init__(self):
        super().__init__('velocity_kinematics_node')

        # --- Geometry (Matches your Part 1 FK) ---
        self.L0 = 36.076
        self.L1 = 60.25
        self.L2z = 128.0
        self.L2x = 24.0
        self.L3 = 124.0
        self.L4 = 133.4

        # Initial state (Will warn if used before robot updates)
        self.current_q = np.zeros(4)
        self.joints_received = False

        # Subscriber: Listen to real robot joints
        self.sub_joints = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_cb, 
            10
        )

        # --- 1a: Forward Velocity Service ---
        self.srv_fk_vel = self.create_service(
            ForwardVelocity, 
            'fk_velocity', 
            self.handle_fk_velocity
        )

        # --- 1b: Inverse Velocity Service ---
        self.srv_ik_vel = self.create_service(
            InverseVelocity, 
            'ik_velocity', 
            self.handle_ik_velocity
        )

        self.get_logger().info("Velocity Kinematics Node Ready.")

    def joint_cb(self, msg):
        # Update q when real data comes in
        self.current_q = np.array(msg.position[:4])
        self.joints_received = True

    def forward_kinematics(self, q):
        """ Helper: Computes Position (x,y,z) for Numerical Jacobian """
        q1, q2, q3, q4 = q
        
        T01 = np.array([[math.cos(q1), -math.sin(q1), 0, 0], [math.sin(q1), math.cos(q1), 0, 0], [0, 0, 1, self.L0], [0, 0, 0, 1]])
        T12 = np.array([[math.cos(q2), 0, math.sin(q2), 0], [0, 1, 0, 0], [-math.sin(q2), 0, math.cos(q2), self.L1], [0, 0, 0, 1]])
        T23 = np.array([[math.cos(q3), 0, math.sin(q3), self.L2x], [0, 1, 0, 0], [-math.sin(q3), 0, math.cos(q3), self.L2z], [0, 0, 0, 1]])
        T34 = np.array([[math.cos(q4), 0, math.sin(q4), self.L3], [0, 1, 0, 0], [-math.sin(q4), 0, math.cos(q4), 0], [0, 0, 0, 1]])
        T4ee = np.array([[math.cos(np.pi/2), 0, math.sin(np.pi/2), self.L4], [0, 1, 0, 0], [-math.sin(np.pi/2), 0, math.cos(np.pi/2), 0], [0, 0, 0, 1]])

        T = T01 @ T12 @ T23 @ T34 @ T4ee
        return T[:3, 3] 

    def compute_jacobian(self, q):
        """ Calculates 3x4 Jacobian numerically """
        eps = 1e-4
        J = np.zeros((3, 4))
        
        for i in range(4):
            q_plus = np.copy(q)
            q_minus = np.copy(q)
            q_plus[i] += eps
            q_minus[i] -= eps
            
            p_plus = self.forward_kinematics(q_plus)
            p_minus = self.forward_kinematics(q_minus)
            
            J[:, i] = (p_plus - p_minus) / (2 * eps)
        return J

    # --- 1a: FK VELOCITY ---
    def handle_fk_velocity(self, request, response):
        # 1. Check for robot state
        if not self.joints_received:
            self.get_logger().warn("No joint states yet. Using q=[0,0,0,0]")

        q_dot = np.array(request.joint_velocities)
        J = self.compute_jacobian(self.current_q)
        v_cartesian = J @ q_dot 
        
        response.end_effector_velocities = v_cartesian.tolist()

        # PRINT TO TERMINAL
        self.get_logger().info(f"> FK REQUEST: Joints Vel {q_dot} -> EE Vel {v_cartesian}")
        
        return response

    # --- 1b: IK VELOCITY ---
    def handle_ik_velocity(self, request, response):
        if not self.joints_received:
            self.get_logger().warn("No joint states yet. Using q=[0,0,0,0]")

        v_cartesian = np.array(request.end_effector_velocities)
        J = self.compute_jacobian(self.current_q)
        J_pinv = np.linalg.pinv(J)
        q_dot = J_pinv @ v_cartesian
        
        response.joint_velocities = q_dot.tolist()

        # PRINT TO TERMINAL
        self.get_logger().info(f"> IK REQUEST: EE Vel {v_cartesian} -> Joints Vel {q_dot}")

        return response

def main(args=None):
    rclpy.init(args=args)
    node = VelocityKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
