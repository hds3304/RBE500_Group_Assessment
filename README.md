# RBE500_Group_Assessment 3

## Usage:

### To Interface with Robot

1. Terminal 1:
    ```bash
    ros2 run GA3 control --ros-args -p Kp:=0.4 -p Kd:=0.0045
    ```

2. Terminal 2:
    ```bash
    ros2 service call /set_j4_reference ga3_srv/srv/J4Control target_position:\ -90.0\ 
    ```

### For Graph

1. To visualise graph between target and actual
    ```bash
    cd src
    python3 plot.py
    ```
