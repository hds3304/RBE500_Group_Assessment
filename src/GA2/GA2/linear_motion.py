#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from ga2_srv.srv import InverseVelocity # The service from Task 1

class LinearMotionTask(Node):
    def __init__(self):
        super().__init__('linear_motion_task')
        
        # Publisher: Send joint velocities to Incremental Node
        self.pub_vel = self.create_publisher(Float64MultiArray, '/joint_velocity_cmd', 10)
        
        # Client: Ask Velocity Kinematics Node for conversions
        self.cli_ik_vel = self.create_client(InverseVelocity, 'ik_velocity')
        
        while not self.cli_ik_vel.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /ik_velocity service...')
            
        self.get_logger().info("Ready to execute Linear Motion Task.")

    def run_task(self):
        # Source 8: "Constant velocity reference in positive y direction"
        # Let's set V_y = 20 mm/s (0.02 m/s). 
        # (Be careful with units! Your FK uses mm, so use 20.0)
        vx = 0.0
        vy = 20.0 
        vz = 0.0
        
        duration = 5.0 # Run for 5 seconds
        rate = 10.0    # 10 Hz (matches sampling_time = 0.1s)
        steps = int(duration * rate)
        
        self.get_logger().info(f"Starting Line Motion: Vy={vy} mm/s for {duration}s")

        for i in range(steps):
            # 1. Prepare Request: Cartesian Vel -> Joint Vel
            req = InverseVelocity.Request()
            req.end_effector_velocities = [vx, vy, vz]
            
            # 2. Call Service (Synchronous for simplicity in this script loop)
            future = self.cli_ik_vel.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            
            q_dot = response.joint_velocities # This is [wd1, wd2, wd3, wd4]
            
            # 3. Publish to Incremental Node
            msg = Float64MultiArray()
            msg.data = q_dot
            self.pub_vel.publish(msg)
            
            self.get_logger().info(f"Step {i+1}/{steps}: Sent vel {q_dot}")
            
            # Sleep to maintain loop rate
            time.sleep(1.0 / rate)
            
        # Stop command (send zeros at the end)
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0, 0.0, 0.0, 0.0]
        self.pub_vel.publish(stop_msg)
        self.get_logger().info("Motion Complete.")

def main(args=None):
    rclpy.init(args=args)
    node = LinearMotionTask()
    try:
        node.run_task()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()