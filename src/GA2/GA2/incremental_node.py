#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from open_manipulator_msgs.srv import SetJointPosition
from std_msgs.msg import Float64MultiArray

class IncrementalNode(Node):
    def __init__(self):
        super().__init__('incremental_node')
        
        # PARAMETER: Sampling time (dt). 
        # Source 5 formula: q_ref = q_old + delta_q * sampling_time
        self.sampling_time = 0.1 # 10Hz (Safety limit)

        # 1. Subscriber: Listen for Joint Velocities (rad/s)
        self.sub_vel = self.create_subscription(
            Float64MultiArray, 
            '/joint_velocity_cmd', 
            self.velocity_cb, 
            10
        )
        
        # 2. Service Client: Send position goals to the robot
        self.cli_robot = self.create_client(SetJointPosition, '/goal_joint_space_path')
        
        # 3. Subscriber: Listen to CURRENT robot joints (for initialization)
        self.sub_joints = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_cb, 
            10
        )
        
        self.q_ref = None  # Will hold [q1, q2, q3, q4]
        self.initialized = False

        self.get_logger().info("Incremental Node Ready. Waiting for joint states...")

    def joint_cb(self, msg):
        """ Initialize q_ref with the robot's ACTUAL position when we start """
        if not self.initialized:
            self.q_ref = list(msg.position[:4])
            self.initialized = True
            self.get_logger().info(f"Initialized q_ref start position: {self.q_ref}")

    def velocity_cb(self, msg):
        """ 
        Received Joint Velocities (rad/s).
        Apply Source 5 Formula: q_ref = q_old + (velocity * dt)
        """
        if not self.initialized:
            self.get_logger().warn("Robot not initialized yet. Ignoring command.")
            return

        # msg.data is [vel_1, vel_2, vel_3, vel_4]
        joint_velocities = msg.data
        
        # Calculate new positions
        new_q_ref = []
        for i in range(4):
            # The integration step
            delta = joint_velocities[i] * self.sampling_time
            self.q_ref[i] += delta
            new_q_ref.append(self.q_ref[i])

        # Send updated q_ref to robot
        self.send_to_robot(new_q_ref)

    def send_to_robot(self, positions):
        if not self.cli_robot.service_is_ready():
            return

        req = SetJointPosition.Request()
        req.planning_group = 'arm'
        req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        # Add 0.0 for gripper
        req.joint_position.position = list(positions) + [0.0]
        # path_time must equal sampling_time for smooth motion
        req.path_time = self.sampling_time 
        
        self.cli_robot.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = IncrementalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()