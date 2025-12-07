#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import os

# Dynamixel SDK imports
from dynamixel_sdk import * 
from ga3_srv.srv import J4Control 

class PDControllerNode(Node):
    def __init__(self):
        super().__init__('pd_controller_node')
        
        # --- CONFIGURATION ---
        # Change based on Dynamixel
        self.DEVICENAME = '/dev/ttyUSB0'
        self.BAUDRATE = 1000000           
        self.PROTOCOL_VERSION = 2.0
        
        # Motor IDs
        self.ID_J1 = 11
        self.ID_J2 = 12
        self.ID_J3 = 13
        self.ID_J4 = 14 
        
        # Control Table Addresses (XM430/XL430)
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_CURRENT = 102
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        
        # [cite_start]PD Gains [cite: 9, 10]
        self.Kp = 3.0   
        self.Kd = 0.05   
        
        # Variables
        self.target_position_j4 = 0   
        self.current_position_j4 = 0  
        self.prev_error = 0
        self.sampling_time = 0.01     # 100Hz
        
        # --- DYNAMIXEL SETUP ---
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        
        if self.portHandler.openPort() and self.portHandler.setBaudRate(self.BAUDRATE):
            self.get_logger().info("Connected to Dynamixel Port")
        else:
            self.get_logger().error("Failed to open port. Check permissions.")
            exit()

        self.setup_robot()

        # --- SERVICE SERVER ---
        self.srv = self.create_service(J4Control, 'set_j4_reference', self.set_ref_cb)
        
        # Timer for Control Loop
        self.timer = self.create_timer(self.sampling_time, self.control_loop)
        
        # [cite_start]Logging [cite: 13]
        self.log_file = open("pd_control_data.csv", "w")
        self.log_file.write("time,target_ticks,actual_ticks,effort\n")
        self.start_time = time.time()

        self.get_logger().info("PD Controller Ready. Input Expected in DEGREES.")

    def write_1byte(self, id, addr, val):
        self.packetHandler.write1ByteTxRx(self.portHandler, id, addr, val)
        
    def write_4byte(self, id, addr, val):
        self.packetHandler.write4ByteTxRx(self.portHandler, id, addr, val)

    def read_4byte(self, id, addr):
        data, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, id, addr)
        if data > 0x7FFFFFFF:
            data -= 4294967296
        return data

    def setup_robot(self):
        # [cite_start]
        self.get_logger().info("Configuring motors[cite: 6]...")
        
        # 1. Disable Torque
        for mid in [self.ID_J1, self.ID_J2, self.ID_J3, self.ID_J4]:
            self.write_1byte(mid, self.ADDR_TORQUE_ENABLE, 0)
            
        # [cite_start]2. Set Operating Modes [cite: 6, 7]
        self.write_1byte(self.ID_J1, self.ADDR_OPERATING_MODE, 3) # Position
        self.write_1byte(self.ID_J2, self.ADDR_OPERATING_MODE, 3) # Position
        self.write_1byte(self.ID_J3, self.ADDR_OPERATING_MODE, 3) # Position
        self.write_1byte(self.ID_J4, self.ADDR_OPERATING_MODE, 0) # Current (Torque)
        
        # 3. Read positions of J1-J3 to hold them
        p1 = self.read_4byte(self.ID_J1, self.ADDR_PRESENT_POSITION)
        p2 = self.read_4byte(self.ID_J2, self.ADDR_PRESENT_POSITION)
        p3 = self.read_4byte(self.ID_J3, self.ADDR_PRESENT_POSITION)
        
        # 4. Enable Torque
        for mid in [self.ID_J1, self.ID_J2, self.ID_J3, self.ID_J4]:
            self.write_1byte(mid, self.ADDR_TORQUE_ENABLE, 1)
            
        # 5. Command J1-J3 to hold
        self.write_4byte(self.ID_J1, self.ADDR_GOAL_POSITION, p1)
        self.write_4byte(self.ID_J2, self.ADDR_GOAL_POSITION, p2)
        self.write_4byte(self.ID_J3, self.ADDR_GOAL_POSITION, p3)
        
        self.current_position_j4 = self.read_4byte(self.ID_J4, self.ADDR_PRESENT_POSITION)
        self.target_position_j4 = self.current_position_j4

    def set_ref_cb(self, request, response):
        # [cite_start]Input is in DEGREES [cite: 8]
        target_deg = request.target_position
        
        # Conversion: Ticks = (deg * 2048 / 180) + 2048
        # Center (0 deg) = 2048 ticks
        self.target_position_j4 = int((target_deg * 2048.0 / 180.0) + 2048.0)
        
        self.get_logger().info(f"Ref Received: {target_deg:.2f} deg -> {self.target_position_j4} ticks")
        response.success = True
        return response

    def control_loop(self):
        # [cite_start]1. Feedback [cite: 8]
        self.current_position_j4 = self.read_4byte(self.ID_J4, self.ADDR_PRESENT_POSITION)
        
        # 2. Error
        error = self.target_position_j4 - self.current_position_j4
        
        # 3. Derivative
        d_error = (error - self.prev_error) / self.sampling_time
        
        # [cite_start]4. PD Law [cite: 3]
        effort = (self.Kp * error) + (self.Kd * d_error)
        
        # 5. Saturate
        MAX_EFFORT = 200 
        effort = max(min(effort, MAX_EFFORT), -MAX_EFFORT)
        
        # [cite_start]6. Actuate [cite: 8]
        self.write_4byte(self.ID_J4, self.ADDR_GOAL_CURRENT, int(effort))
        
        # 7. Update
        self.prev_error = error
        
        # [cite_start]8. Log [cite: 13]
        t = time.time() - self.start_time
        self.log_file.write(f"{t},{self.target_position_j4},{self.current_position_j4},{effort}\n")

def main(args=None):
    rclpy.init(args=args)
    node = PDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_1byte(node.ID_J4, node.ADDR_TORQUE_ENABLE, 0)
        node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()