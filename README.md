# RBE500_Group_Assessment

GitHub Run Commands fo FK.py

Terminal 1: ros2 run GA1 FK
Terminal 2: ros2 topic echo /end_effector_pose & ros2 topic pub --once /joint_states sensor_msgs/JointState -- "{name: ['joint1','joint2','joint3', 'joint4'], position: [0.0, 0.0, 0.0, 0.0]}"
