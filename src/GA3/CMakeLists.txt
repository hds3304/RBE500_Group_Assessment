cmake_minimum_required(VERSION 3.8)
project(GA3)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(ga3_srv REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(control src/control.cpp)
ament_target_dependencies(control
  rclcpp
  ga3_srv
  dynamixel_sdk
)

install(TARGETS control
  DESTINATION lib/${PROJECT_NAME})

ament_package()
