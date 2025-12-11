#include <chrono>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include "ga3_srv/srv/j4_control.hpp"

// Dynamixel SDK
#include "dynamixel_sdk/dynamixel_sdk.h"

using namespace std::chrono_literals;
using dynamixel::PortHandler;
using dynamixel::PacketHandler;

class PDControllerNode : public rclcpp::Node
{
public:
    PDControllerNode() : Node("pd_controller_node")
    {
        // --- CONFIG ---
        DEVICENAME = "/dev/ttyUSB0";
        BAUDRATE = 1000000;
        PROTOCOL_VERSION = 2.0;

        ID_J1 = 11;
        ID_J2 = 12;
        ID_J3 = 13;
        ID_J4 = 14;

        ADDR_OPERATING_MODE  = 11;
        ADDR_TORQUE_ENABLE   = 64;
        ADDR_GOAL_CURRENT    = 102;
        ADDR_GOAL_POSITION   = 116;
        ADDR_PRESENT_POSITION = 132;

        declare_parameter<double>("Kp", 0.4);   // default value
        declare_parameter<double>("Kd", 0.0045);  // default value

        Kp = get_parameter("Kp").as_double();
        Kd = get_parameter("Kd").as_double();

        RCLCPP_INFO(get_logger(), "Using Kp = %.3f, Kd = %.3f", Kp, Kd);

        // target_position_j4 = 0;
        // current_position_j4 = 0;
        prev_error = 0;
        sampling_time = 0.01;

        // --- Dynamixel Setup ---
        portHandler = PortHandler::getPortHandler(DEVICENAME.c_str());
        packetHandler = PacketHandler::getPacketHandler(PROTOCOL_VERSION);

        if (portHandler->openPort() && portHandler->setBaudRate(BAUDRATE)) {
            RCLCPP_INFO(get_logger(), "Connected to Dynamixel Port");
        } else {
            RCLCPP_ERROR(get_logger(), "Failed to open Dynamixel port");
            rclcpp::shutdown();
        }

        setup_robot();

        // --- Service Server ---
        srv_ = create_service<ga3_srv::srv::J4Control>(
            "set_j4_reference",
            std::bind(&PDControllerNode::set_ref_cb, this,
                      std::placeholders::_1, std::placeholders::_2)
        );

        // --- Timer ---
        timer_ = create_wall_timer(
            10ms, std::bind(&PDControllerNode::control_loop, this)
        );

        // --- Logging ---
        log_file.open("pd_control_data.csv");
        log_file << "time,target_ticks,actual_ticks,effort\n";
        start_time = now_sec();

        RCLCPP_INFO(get_logger(), "PD Controller READY. Input in DEGREES.");
    }

    ~PDControllerNode()
    {
        write1(ID_J4, ADDR_TORQUE_ENABLE, 0);
        log_file.close();
        portHandler->closePort();
    }

private:

    // ---------------------------------------------------
    // --- Utility Functions ---
    // ---------------------------------------------------

    double now_sec()
    {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void write1(int id, int addr, int val)
    {
        uint8_t dxl_error;
        packetHandler->write1ByteTxRx(portHandler, id, addr, val, &dxl_error);
    }

    int read1(int id, int addr)
    {
        uint8_t data = 0;
        uint8_t dxl_error = 0;
        packetHandler->read1ByteTxRx(portHandler, id, addr, &data, &dxl_error);
        return data;
    }

    void write2(int id, int addr, int16_t value)
    {
        uint8_t dxl_error = 0;

        int comm_result = packetHandler->write2ByteTxRx(
            portHandler,
            id,
            addr,
            value,
            &dxl_error
        );

        if (comm_result != COMM_SUCCESS) {
            RCLCPP_ERROR(get_logger(), 
                "WRITE2 FAILED: %s", 
                packetHandler->getTxRxResult(comm_result)
            );
        }

        if (dxl_error != 0) {
            RCLCPP_ERROR(get_logger(), 
                "DYNAMIXEL ERROR: %s", 
                packetHandler->getRxPacketError(dxl_error)
            );
        }
    }

    int read2(int id, int addr)
    {
        uint16_t data = 0;
        uint8_t dxl_error = 0;

        int result = packetHandler->read2ByteTxRx(
            portHandler,
            id,
            addr,
            &data,
            &dxl_error
        );

        return data;   // returns 0–65535
    }

    void write4(int id, int addr, int val)
    {
        uint8_t dxl_error;
        packetHandler->write4ByteTxRx(portHandler, id, addr, val, &dxl_error);
    }

    int read4(int id, int addr)
    {
        uint32_t data = 0;
        uint8_t dxl_error;
        packetHandler->read4ByteTxRx(portHandler, id, addr, &data, &dxl_error);

        if (data > 0x7FFFFFFF)
            data -= 4294967296;  // convert unsigned → signed

        return static_cast<int>(data);
    }

    double ticksToDegrees(int ticks, int center = 2048)
    {
        // Convert ticks to degrees
        double deg = (static_cast<double>(ticks) - center) * 180.0 / 2048.0;
        return std::round(deg);
    }

    // ---------------------------------------------------
    // --- Setup Motors ---
    // ---------------------------------------------------
    void setup_robot()
    {
        RCLCPP_INFO(get_logger(), "Configuring motors...");

        // 1. Disable Torque
        for (int id : {ID_J1, ID_J2, ID_J3, ID_J4})
            write1(id, ADDR_TORQUE_ENABLE, 0);

        // 2. Set Operating Modes
        write1(ID_J1, ADDR_OPERATING_MODE, 3);
        write1(ID_J2, ADDR_OPERATING_MODE, 3);
        write1(ID_J3, ADDR_OPERATING_MODE, 3);
        write1(ID_J4, ADDR_OPERATING_MODE, 0);   // current mode

        // 3. Read and hold J1–J3
        // int p1 = read4(ID_J1, ADDR_PRESENT_POSITION);
        // int p2 = read4(ID_J2, ADDR_PRESENT_POSITION);
        // int p3 = read4(ID_J3, ADDR_PRESENT_POSITION);

        int p1 = 2048;
        int p2 = 2048;
        int p3 = 2048;

        // 4. Enable torque
        for (int id : {ID_J1, ID_J2, ID_J3, ID_J4})
            write1(id, ADDR_TORQUE_ENABLE, 1);

        // 5. Hold positions
        write4(ID_J1, ADDR_GOAL_POSITION, p1);
        write4(ID_J2, ADDR_GOAL_POSITION, p2);
        write4(ID_J3, ADDR_GOAL_POSITION, p3);

        current_position_j4 = read4(ID_J4, ADDR_PRESENT_POSITION);
        target_position_j4  = current_position_j4;
    }

    // ---------------------------------------------------
    // --- Service Callback ---
    // ---------------------------------------------------
    void set_ref_cb(
        const std::shared_ptr<ga3_srv::srv::J4Control::Request> req,
        std::shared_ptr<ga3_srv::srv::J4Control::Response> res)
    {
        double target_deg = req->target_position;

        // ticks = (deg * 2048 / 180) + 2048
        target_position_j4 = static_cast<int>((target_deg * 2048.0 / 180.0) + 2048.0);

        RCLCPP_INFO(get_logger(),
                    "Received %.2f deg -> %d ticks",
                    target_deg, target_position_j4);

        res->success = true;
    }

    // ---------------------------------------------------
    // --- Main Control Loop ---
    // ---------------------------------------------------
    void control_loop()
    {
        // 1. Feedback
        current_position_j4 = read4(ID_J4, ADDR_PRESENT_POSITION);


        // 2. Error
        double error = target_position_j4 - current_position_j4;

        // 3. Derivative
        double d_error = (error - prev_error) / sampling_time;

        // 4. PD Output
        double effort = (Kp * error) + (Kd * d_error);

        // 5. Saturate
        const double MAX_EFFORT = 300;
        effort = std::clamp(effort, -MAX_EFFORT, MAX_EFFORT);

        // 6. Actuate
        // write4(ID_J4, ADDR_GOAL_CURRENT, static_cast<int>(effort));
        write2(ID_J4, ADDR_GOAL_CURRENT, (int16_t)effort);

        // 7. Update
        prev_error = error;

        // 8. Logging
        RCLCPP_INFO(get_logger(),
            "Current Posn %d\n"
            "Error %.2f\n"
            "D Error %.2f\n"
            "Effort %.2f\n",
            current_position_j4, error, d_error, effort);
        
        int op_mode = read1(ID_J4, ADDR_OPERATING_MODE);
        RCLCPP_INFO(get_logger(), "KP = %f | KD = %f", Kp, Kd);

        double t = now_sec() - start_time;
        log_file << t << "," << ticksToDegrees(target_position_j4) << ","
                 << ticksToDegrees(current_position_j4) << "," << effort << "\n";
    }

    // ---------------------------------------------------
    // --- Members ---
    // ---------------------------------------------------
    std::string DEVICENAME;
    int BAUDRATE;
    double PROTOCOL_VERSION;

    int ID_J1, ID_J2, ID_J3, ID_J4;

    int ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE;
    int ADDR_GOAL_CURRENT, ADDR_GOAL_POSITION;
    int ADDR_PRESENT_POSITION;

    double Kp, Kd;
    int target_position_j4, current_position_j4;
    double prev_error;
    double sampling_time;

    PortHandler *portHandler;
    PacketHandler *packetHandler;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<ga3_srv::srv::J4Control>::SharedPtr srv_;

    std::ofstream log_file;
    double start_time;
};

// ---------------------------------------------------
// --- main() ---
// ---------------------------------------------------
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PDControllerNode>();

    auto start = std::chrono::steady_clock::now();

    // Run for 10 seconds
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();

        if (elapsed >= 10.0) {
            RCLCPP_INFO(node->get_logger(), "10 seconds elapsed — shutting down.");
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    rclcpp::shutdown();
    return 0;
}
