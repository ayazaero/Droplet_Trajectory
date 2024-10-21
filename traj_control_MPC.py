'''Similar to traj_control.py, this script defines a ROS node for controlling a vehicle using MPC.
It focuses on offboard control for takeoff, landing, and trajectory following with position 
and velocity setpoints based on MPC outputs.'''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
import numpy as np
import cvxpy as cp
import time


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/vehicle_command_control', qos_profile) #fmu/in/Trajectory_setpoint
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = 10
        self.takeoff_X = 0
        self.takeoff_Y = 0
        self.takeoff_counter = 0

        self.entry_point = 1.5*np.array([2.0, 2.0, 2.0])
        self.destination = np.array([0.0, 0.0, 0.7])
        self.control_mode = 'takeoff'  # Modes: 'position', 'MPC', 'landing'

        self.pos=True
        self.vel=False


        # MPC parameters
        self.T = 5  # horizon length
        self.nx = 3  # number of state variables
        self.nu = 3  # number of control inputs
        self.dt = 0.1  # time step
        self.Q = np.eye(self.nx)  # state cost matrix
        self.R = np.eye(self.nu)  # control cost matrix

        # MPC variables setup
        self.x1 = cp.Variable((self.T+1, self.nx))
        self.u1 = cp.Variable((self.T, self.nu))


        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = self.pos
        msg.velocity = self.vel
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, -z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        #self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_velocity_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [float('nan')] * 3
        msg.velocity = [x, y, -z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing velocity setpoints {[x, y, z]}")

    def publish_takeoff(self):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [0.0, 0.0, -0.5]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Taking Off!")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def initialize_mpc(self, initial_position):
        """Setup and solve the MPC to compute the initial trajectory."""
        constraints = [self.x1[0] == initial_position]
        constraints += [self.x1[k+1] == self.x1[k] + self.dt * self.u1[k] for k in range(self.T)]
        constraints += [cp.norm(self.u1[k], 2) <= 0.5 for k in range(self.T)]

        objective = cp.Minimize(sum(cp.quad_form(self.x1[k]-self.destination, self.Q) + cp.quad_form(self.u1[k], self.R) for k in range(self.T)))
        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        # Update control inputs and state predictions
        self.control_inputs = self.u1.value
        self.state_predictions = self.x1.value

    def mpc_control(self):
        """Get the optimal control input from MPC."""
        if self.control_inputs is not None:
            return self.control_inputs[0]  # Return the first control action
        else:
            return np.zeros(self.nu)  # Default control if MPC not solved


    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()
        self.get_logger().info(self.control_mode)


        current_pos = np.array([self.vehicle_local_position.x,
                            self.vehicle_local_position.y,
                            -self.vehicle_local_position.z])
        
        if self.offboard_setpoint_counter == 20:
            self.engage_offboard_mode()
            self.arm()

       
        if self.control_mode == 'takeoff' and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_takeoff()
            self.takeoff_counter += 1
            if self.takeoff_counter >10:
                self.control_mode = 'position'
        elif self.control_mode == 'position' and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_position_setpoint(self.entry_point[0],self.entry_point[1],self.entry_point[2])
            if np.linalg.norm(current_pos - self.entry_point) < 0.3:
                self.control_mode = 'MPC'
                # Initialize MPC with current state
                self.initialize_mpc(current_pos)
                self.pos=False
                self.vel = True
        elif self.control_mode == 'MPC' and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.initialize_mpc(current_pos)
            self.publish_velocity_setpoint(*self.mpc_control())
            current_pos = np.array([
                self.vehicle_local_position.x,
                self.vehicle_local_position.y,
                -self.vehicle_local_position.z
            ])
            if np.linalg.norm(current_pos - self.destination) < 0.1 or current_pos[2]<0.5:
                self.control_mode = 'landing'
                self.land()
                exit(0)



        if self.offboard_setpoint_counter < 21:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)