import sys
import time
import rclpy
from rclpy.node import Node
from fkik_srv.srv import InvKinService


class IKClient(Node):
    def __init__(self):
        super().__init__('ik_client')

        self.client = self.create_client(InvKinService, '/get_ik')
        while not self.client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().error('Interrupted while waiting for the service. Exiting.')
                sys.exit(0)
            self.get_logger().info('Service not available, waiting again...')

        self.get_logger().info('IK Client ready.')

    def send_goal(self, pose_values, gripper_orientation, gripper_value):
        self.get_logger().info(f'Sending robot to: pos={pose_values}, ori={gripper_orientation}, grip={gripper_value}')

        request = InvKinService.Request()
        request.pose_values.position.x = float(pose_values[0])
        request.pose_values.position.y = float(pose_values[1])
        request.pose_values.position.z = float(pose_values[2])
        request.gripper_orientation = gripper_orientation
        request.gripper_value = gripper_value

        self.client.call_async(request)
        self.get_logger().info('Goal sent.')


def main(args=None):
    rclpy.init(args=args)
    node = IKClient()

    commands = [
        [(250.0, -180.0, 281.0), 90.0, 0.01],
        [(250.0, -180.0, 210.0), 75.0, -0.01],
        [(250.0, -180.0, 281.0), 90.0, -0.01],
        [(250.0, 180.0, 281.0), 90.0, -0.01],
        [(250.0, 180.0, 210.0), 75.0, 0.01],
        [(250.0, 180.0, 281.0), 90.0, 0.01],
        [(224.0, 0.0, 281.0), 0.0, -0.01]
    ]

    print("Bark")  # This will now execute

    for command in commands:
        print(f"Executing command: {command}")
        try:
            node.send_goal(*command)  # Now send_goal will actually run
            time.sleep(5)
        except Exception as e:
            node.get_logger().error(f'Service call failed: {e}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
