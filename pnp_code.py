import cv2
import numpy as np
import time
import threading
import math
import logging
from typing import Tuple, List, Optional, Dict
from enum import Enum
from dataclasses import dataclass
import heapq
import json
import serial
import struct

# Hardware interface libraries
try:
    import Jetson.GPIO as GPIO
    import pyrealsense2 as rs
    from pymodbus.client.sync import ModbusSerialClient as ModbusClient
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware libraries not available: {e}")
    HARDWARE_AVAILABLE = False
    # Mock GPIO for development
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        HIGH = 1
        LOW = 0
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setup(pin, mode): pass
        @staticmethod
        def output(pin, value): pass
        @staticmethod
        def input(pin): return 0
        @staticmethod
        def cleanup(): pass
    GPIO = MockGPIO()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobotState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    PLANNING = "planning"
    MOVING = "moving"
    PICKING = "picking"
    PLACING = "placing"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class Position3D:
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Position3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class DetectedObject:
    position: Position3D
    dimensions: Tuple[float, float, float]
    confidence: float
    object_type: str
    color: str
    depth_data: Optional[np.ndarray] = None

@dataclass
class JointAngles:
    base: float
    shoulder: float
    elbow: float
    wrist: float

class DeltaServoController:
    """Hardware interface for Delta ASD-A2 servo drivers"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.client = None
        self.servo_addresses = [1, 2, 3, 4]  # Modbus addresses for 4 servos
        
        # Servo parameters (adjust based on your specific motors)
        self.encoder_resolution = 131072  # 17-bit encoder
        self.gear_ratios = [100, 50, 50, 30]  # Gear ratios for each joint
        
        # Current positions (encoder counts)
        self.current_positions = [0, 0, 0, 0]
        self.target_positions = [0, 0, 0, 0]
        
        # Safety limits (in degrees)
        self.joint_limits = [
            (-180, 180),  # Base
            (-90, 90),    # Shoulder
            (-135, 135),  # Elbow
            (-180, 180)   # Wrist
        ]
        
        self._initialize_modbus()
    
    def _initialize_modbus(self):
        """Initialize Modbus communication"""
        try:
            if HARDWARE_AVAILABLE:
                self.client = ModbusClient(
                    method='rtu',
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=3,
                    parity='N',
                    stopbits=1,
                    bytesize=8
                )
                
                if self.client.connect():
                    logger.info("Modbus connection established")
                    self._configure_servos()
                    return True
                else:
                    logger.error("Failed to connect to Modbus")
                    return False
            else:
                logger.warning("Running in simulation mode - no hardware control")
                return True
        except Exception as e:
            logger.error(f"Modbus initialization error: {e}")
            return False
    
    def _configure_servos(self):
        """Configure servo parameters"""
        try:
            for i, addr in enumerate(self.servo_addresses):
                if self.client:
                    # Set servo to position control mode
                    self.client.write_register(0x2000, 1, unit=addr)
                    
                    # Set position loop gain parameters
                    self.client.write_register(0x2001, 100, unit=addr)  # P gain
                    self.client.write_register(0x2002, 10, unit=addr)   # I gain
                    self.client.write_register(0x2003, 5, unit=addr)    # D gain
                    
                    # Enable servo
                    self.client.write_register(0x2004, 1, unit=addr)
            
            logger.info("Servo configuration completed")
        except Exception as e:
            logger.error(f"Servo configuration error: {e}")
    
    def degrees_to_encoder(self, degrees: float, joint_index: int) -> int:
        """Convert degrees to encoder counts"""
        gear_ratio = self.gear_ratios[joint_index]
        encoder_counts = int((degrees / 360.0) * self.encoder_resolution * gear_ratio)
        return encoder_counts
    
    def encoder_to_degrees(self, encoder_counts: int, joint_index: int) -> float:
        """Convert encoder counts to degrees"""
        gear_ratio = self.gear_ratios[joint_index]
        degrees = (encoder_counts / (self.encoder_resolution * gear_ratio)) * 360.0
        return degrees
    
    def move_joint(self, joint_index: int, target_degrees: float, max_speed: float = 30.0) -> bool:
        """Move single joint to target position"""
        try:
            # Check joint limits
            min_angle, max_angle = self.joint_limits[joint_index]
            if not (min_angle <= target_degrees <= max_angle):
                logger.error(f"Joint {joint_index} target {target_degrees}° exceeds limits ({min_angle}, {max_angle})")
                return False
            
            # Convert to encoder counts
            target_encoder = self.degrees_to_encoder(target_degrees, joint_index)
            addr = self.servo_addresses[joint_index]
            
            if self.client:
                # Set target position
                self.client.write_register(0x2010, target_encoder & 0xFFFF, unit=addr)
                self.client.write_register(0x2011, (target_encoder >> 16) & 0xFFFF, unit=addr)
                
                # Set speed limit
                speed_encoder = int((max_speed / 360.0) * self.encoder_resolution * self.gear_ratios[joint_index] / 60.0)
                self.client.write_register(0x2012, speed_encoder, unit=addr)
                
                # Start motion
                self.client.write_register(0x2013, 1, unit=addr)
            else:
                # Simulation mode
                time.sleep(abs(target_degrees - self.encoder_to_degrees(self.current_positions[joint_index], joint_index)) / max_speed)
            
            self.target_positions[joint_index] = target_encoder
            logger.info(f"Joint {joint_index} moving to {target_degrees:.1f}°")
            return True
            
        except Exception as e:
            logger.error(f"Joint motion error: {e}")
            return False
    
    def move_all_joints(self, joint_angles: JointAngles, max_speed: float = 20.0) -> bool:
        """Move all joints simultaneously"""
        try:
            angles = [joint_angles.base, joint_angles.shoulder, joint_angles.elbow, joint_angles.wrist]
            
            # Check all joint limits first
            for i, angle in enumerate(angles):
                min_angle, max_angle = self.joint_limits[i]
                if not (min_angle <= angle <= max_angle):
                    logger.error(f"Joint {i} target {angle}° exceeds limits")
                    return False
            
            # Start all motions simultaneously
            for i, angle in enumerate(angles):
                if not self.move_joint(i, angle, max_speed):
                    logger.error(f"Failed to start motion for joint {i}")
                    return False
            
            # Wait for all motions to complete
            return self.wait_for_motion_complete(timeout=10.0)
            
        except Exception as e:
            logger.error(f"Multi-joint motion error: {e}")
            return False
    
    def wait_for_motion_complete(self, timeout: float = 5.0) -> bool:
        """Wait for all servo motions to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_complete = True
            
            for i, addr in enumerate(self.servo_addresses):
                try:
                    if self.client:
                        # Read motion status register
                        result = self.client.read_holding_registers(0x2020, 1, unit=addr)
                        if result.isError() or result.registers[0] != 0:
                            all_complete = False
                            break
                        
                        # Update current position
                        pos_low = self.client.read_holding_registers(0x2021, 1, unit=addr).registers[0]
                        pos_high = self.client.read_holding_registers(0x2022, 1, unit=addr).registers[0]
                        self.current_positions[i] = pos_low | (pos_high << 16)
                    else:
                        # Simulation mode
                        self.current_positions[i] = self.target_positions[i]
                except Exception as e:
                    logger.error(f"Status read error for joint {i}: {e}")
                    all_complete = False
                    break
            
            if all_complete:
                logger.info("All joint motions completed")
                return True
            
            time.sleep(0.1)
        
        logger.warning("Motion timeout - joints may not have reached target")
        return False
    
    def get_current_angles(self) -> JointAngles:
        """Get current joint angles"""
        angles = []
        for i in range(4):
            degrees = self.encoder_to_degrees(self.current_positions[i], i)
            angles.append(degrees)
        
        return JointAngles(angles[0], angles[1], angles[2], angles[3])
    
    def emergency_stop(self):
        """Emergency stop all servos"""
        try:
            for addr in self.servo_addresses:
                if self.client:
                    self.client.write_register(0x2030, 1, unit=addr)  # Emergency stop command
            logger.critical("EMERGENCY STOP ACTIVATED")
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")
    
    def enable_servos(self, enable: bool = True):
        """Enable or disable all servos"""
        try:
            command = 1 if enable else 0
            for addr in self.servo_addresses:
                if self.client:
                    self.client.write_register(0x2004, command, unit=addr)
            logger.info(f"Servos {'enabled' if enable else 'disabled'}")
        except Exception as e:
            logger.error(f"Servo enable/disable error: {e}")

class RealSenseVisionSystem:
    """Intel RealSense D435i camera interface"""
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.is_streaming = False
        
        # Detection parameters
        self.detection_params = {
            'min_depth': 200,      # mm
            'max_depth': 1500,     # mm
            'min_area': 500,       # pixels
            'max_area': 50000,     # pixels
            'depth_threshold': 10   # mm depth difference
        }
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize RealSense camera"""
        try:
            if HARDWARE_AVAILABLE:
                import pyrealsense2 as rs
                
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                
                # Configure streams
                self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                
                # Start streaming
                self.pipeline.start(self.config)
                self.is_streaming = True
                
                # Create alignment object
                self.align = rs.align(rs.stream.color)
                
                logger.info("RealSense camera initialized successfully")
            else:
                logger.warning("RealSense not available - using simulation")
                
        except Exception as e:
            logger.error(f"RealSense initialization error: {e}")
    
    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture aligned color and depth frames"""
        try:
            if not self.is_streaming or not HARDWARE_AVAILABLE:
                # Return simulated frames for testing
                color_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                depth_frame = np.full((720, 1280), 500, dtype=np.uint16)  # 500mm depth
                return color_frame, depth_frame
            
            import pyrealsense2 as rs
            
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None, None
    
    def detect_objects_3d(self) -> List[DetectedObject]:
        """Detect objects using RGB-D data"""
        try:
            color_frame, depth_frame = self.capture_frame()
            
            if color_frame is None or depth_frame is None:
                return []
            
            # Convert color to HSV for better object detection
            hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
            
            # Create depth mask to filter objects within working range
            depth_mask = cv2.inRange(depth_frame, 
                                   self.detection_params['min_depth'],
                                   self.detection_params['max_depth'])
            
            detected_objects = []
            
            # Define color ranges for object classification
            color_ranges = {
                'red': ((0, 100, 100), (10, 255, 255)),
                'blue': ((100, 100, 100), (130, 255, 255)),
                'green': ((50, 100, 100), (70, 255, 255)),
                'yellow': ((20, 100, 100), (30, 255, 255))
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                # Create color mask
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Combine with depth mask
                combined_mask = cv2.bitwise_and(color_mask, depth_mask)
                
                # Clean up mask
                kernel = np.ones((5, 5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if self.detection_params['min_area'] <= area <= self.detection_params['max_area']:
                        obj = self._analyze_3d_object(contour, color_name, color_frame, depth_frame)
                        if obj:
                            detected_objects.append(obj)
            
            logger.info(f"Detected {len(detected_objects)} 3D objects")
            return detected_objects
            
        except Exception as e:
            logger.error(f"3D object detection error: {e}")
            return []
    
    def _analyze_3d_object(self, contour: np.ndarray, color: str, 
                          color_frame: np.ndarray, depth_frame: np.ndarray) -> Optional[DetectedObject]:
        """Analyze contour to extract 3D object information"""
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract depth data for object region
            object_depth_region = depth_frame[y:y+h, x:x+w]
            
            # Filter out zero/invalid depth values
            valid_depths = object_depth_region[object_depth_region > 0]
            
            if len(valid_depths) < 10:  # Need minimum valid depth points
                return None
            
            # Calculate object depth statistics
            median_depth = np.median(valid_depths)
            depth_std = np.std(valid_depths)
            
            # Skip if depth variation is too high (likely multiple objects)
            if depth_std > 50:  # 50mm threshold
                return None
            
            # Convert pixel coordinates to 3D world coordinates
            center_x, center_y = x + w//2, y + h//2
            world_pos = self._pixel_to_world_3d(center_x, center_y, median_depth)
            
            # Estimate 3D dimensions using depth data
            dimensions = self._estimate_3d_dimensions(object_depth_region, median_depth)
            
            # Calculate confidence based on depth consistency and size
            confidence = self._calculate_3d_confidence(valid_depths, area=w*h)
            
            # Determine object type
            object_type = self._classify_3d_object(dimensions, color, confidence)
            
            return DetectedObject(
                position=Position3D(world_pos[0], world_pos[1], world_pos[2]),
                dimensions=dimensions,
                confidence=confidence,
                object_type=object_type,
                color=color,
                depth_data=object_depth_region
            )
            
        except Exception as e:
            logger.error(f"3D object analysis error: {e}")
            return None
    
    def _pixel_to_world_3d(self, pixel_x: int, pixel_y: int, depth_mm: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates and depth to 3D world coordinates"""
        # RealSense D435i intrinsic parameters (approximate)
        fx, fy = 615.0, 615.0  # Focal lengths
        cx, cy = 640.0, 360.0  # Principal point
        
        # Convert to meters
        depth_m = depth_mm / 1000.0
        
        # Calculate 3D coordinates in camera frame
        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m
        
        # Transform to robot base frame (adjust based on camera mounting)
        # Assuming camera is mounted above workspace looking down
        x_world = x_cam * 1000  # Convert back to mm
        y_world = -y_cam * 1000  # Flip Y axis
        z_world = 400 - z_cam * 1000  # Height from base, adjust based on mounting
        
        return x_world, y_world, z_world
    
    def _estimate_3d_dimensions(self, depth_region: np.ndarray, median_depth: float) -> Tuple[float, float, float]:
        """Estimate 3D object dimensions from depth data"""
        try:
            # Get object mask (pixels close to median depth)
            depth_threshold = self.detection_params['depth_threshold']
            object_mask = np.abs(depth_region - median_depth) < depth_threshold
            
            if np.sum(object_mask) < 5:
                return 40.0, 40.0, 20.0  # Default dimensions
            
            # Find object bounds in depth image
            rows, cols = np.where(object_mask)
            
            if len(rows) == 0 or len(cols) == 0:
                return 40.0, 40.0, 20.0
            
            # Calculate pixel dimensions
            width_pixels = np.max(cols) - np.min(cols) + 1
            height_pixels = np.max(rows) - np.min(rows) + 1
            
            # Convert to real-world dimensions using depth
            # Approximate conversion: 1 pixel ≈ depth(mm) / 1000 mm at 1m distance
            pixel_size = median_depth / 1000.0  # mm per pixel at current depth
            
            width_mm = width_pixels * pixel_size
            height_mm = height_pixels * pixel_size
            
            # Estimate depth/thickness from depth variation
            valid_depths = depth_region[object_mask]
            depth_range = np.max(valid_depths) - np.min(valid_depths)
            thickness_mm = max(10.0, min(50.0, depth_range))  # Clamp between 10-50mm
            
            # Ensure minimum dimensions
            width_mm = max(20.0, width_mm)
            height_mm = max(20.0, height_mm)
            
            return width_mm, height_mm, thickness_mm
            
        except Exception as e:
            logger.error(f"3D dimension estimation error: {e}")
            return 40.0, 40.0, 20.0
    
    def _calculate_3d_confidence(self, depth_values: np.ndarray, area: int) -> float:
        """Calculate detection confidence based on 3D data quality"""
        try:
            # Depth consistency score
            depth_std = np.std(depth_values)
            consistency_score = max(0.0, 1.0 - depth_std / 100.0)  # Normalize by 100mm
            
            # Size score (prefer medium-sized objects)
            ideal_area = 2000  # pixels
            size_score = 1.0 - abs(area - ideal_area) / ideal_area
            size_score = max(0.0, min(1.0, size_score))
            
            # Number of valid depth points score
            num_points = len(depth_values)
            points_score = min(1.0, num_points / 100.0)
            
            # Weighted combination
            confidence = 0.5 * consistency_score + 0.3 * size_score + 0.2 * points_score
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"3D confidence calculation error: {e}")
            return 0.5
    
    def _classify_3d_object(self, dimensions: Tuple[float, float, float], 
                           color: str, confidence: float) -> str:
        """Classify object based on 3D properties"""
        width, height, thickness = dimensions
        volume = width * height * thickness
        
        if volume < 20000:  # Small objects
            return "small_part"
        elif volume < 100000:  # Medium objects
            return "medium_part"
        else:
            return "large_part"
    
    def stop_streaming(self):
        """Stop camera streaming"""
        try:
            if self.pipeline and self.is_streaming:
                self.pipeline.stop()
                self.is_streaming = False
                logger.info("RealSense streaming stopped")
        except Exception as e:
            logger.error(f"Camera stop error: {e}")

class SensorInterface:
    """Interface for proximity and torque sensors"""
    
    def __init__(self):
        # GPIO pin assignments (based on your wiring diagram)
        self.proximity_sensor_pin = 18
        self.torque_sensor_can_pin = 19  # CAN interface pin
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.proximity_sensor_pin, GPIO.IN)
        
        # CAN bus for torque sensor (simplified - you may need python-can library)
        self.torque_value = 0.0
        
        logger.info("Sensor interface initialized")
    
    def read_proximity_sensor(self) -> bool:
        """Read proximity sensor state"""
        try:
            return bool(GPIO.input(self.proximity_sensor_pin))
        except Exception as e:
            logger.error(f"Proximity sensor read error: {e}")
            return False
    
    def read_torque_sensor(self) -> float:
        """Read torque sensor value via CAN bus"""
        try:
            # Simplified torque reading - implement actual CAN communication
            # For now, simulate based on gripper state
            if hasattr(self, '_simulated_torque'):
                return self._simulated_torque
            return 0.0
        except Exception as e:
            logger.error(f"Torque sensor read error: {e}")
            return 0.0
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        try:
            GPIO.cleanup()
        except Exception as e:
            logger.error(f"GPIO cleanup error: {e}")

class VacuumGripperSystem:
    """Vacuum gripper control system"""
    
    def __init__(self, sensor_interface: SensorInterface):
        self.sensor_interface = sensor_interface
        
        # GPIO pins for gripper control
        self.vacuum_pump_pin = 12
        self.vacuum_valve_pin = 16
        
        # Setup GPIO
        GPIO.setup(self.vacuum_pump_pin, GPIO.OUT)
        GPIO.setup(self.vacuum_valve_pin, GPIO.OUT)
        
        # Initial state
        self.vacuum_active = False
        self.is_gripping = False
        self.grip_pressure = 0.0
        
        # Turn off vacuum initially
        GPIO.output(self.vacuum_pump_pin, GPIO.LOW)
        GPIO.output(self.vacuum_valve_pin, GPIO.LOW)
        
        logger.info("Vacuum gripper system initialized")
    
    def activate_vacuum(self) -> bool:
        """Activate vacuum pump and valve"""
        try:
            GPIO.output(self.vacuum_pump_pin, GPIO.HIGH)
            GPIO.output(self.vacuum_valve_pin, GPIO.HIGH)
            self.vacuum_active = True
            
            time.sleep(0.5)  # Allow vacuum to build
            
            logger.info("Vacuum activated")
            return True
            
        except Exception as e:
            logger.error(f"Vacuum activation error: {e}")
            return False
    
    def deactivate_vacuum(self) -> bool:
        """Deactivate vacuum system"""
        try:
            GPIO.output(self.vacuum_valve_pin, GPIO.LOW)
            GPIO.output(self.vacuum_pump_pin, GPIO.LOW)
            self.vacuum_active = False
            self.is_gripping = False
            self.grip_pressure = 0.0
            
            logger.info("Vacuum deactivated")
            return True
            
        except Exception as e:
            logger.error(f"Vacuum deactivation error: {e}")
            return False
    
    def attempt_grip(self) -> bool:
        """Attempt to grip object"""
        try:
            # Check if object is in range using proximity sensor
            if not self.sensor_interface.read_proximity_sensor():
                logger.warning("No object detected for gripping")
                return False
            
            # Activate vacuum
            if not self.activate_vacuum():
                return False
            
            # Wait and check if grip is successful
            time.sleep(1.0)
            
            # Check torque sensor for grip confirmation
            torque = self.sensor_interface.read_torque_sensor()
            
            if torque > 0.5:  # Threshold for successful grip (N⋅m)
                self.is_gripping = True
                self.grip_pressure = torque * 10  # Convert to approximate pressure
                logger.info(f"Object gripped successfully, torque: {torque:.2f} N⋅m")
                return True
            else:
                # Failed to grip - release vacuum
                self.deactivate_vacuum()
                logger.warning("Failed to establish grip")
                return False
                
        except Exception as e:
            logger.error(f"Grip attempt error: {e}")
            return False
    
    def release_grip(self) -> bool:
        """Release gripped object"""
        try:
            return self.deactivate_vacuum()
        except Exception as e:
            logger.error(f"Grip release error: {e}")
            return False
    
    def monitor_grip_integrity(self) -> bool:
        """Monitor grip integrity during motion"""
        if not self.is_gripping:
            return True
        
        try:
            # Check proximity sensor
            if not self.sensor_interface.read_proximity_sensor():
                logger.critical("GRIP FAILURE - Object not detected!")
                self.is_gripping = False
                return False
            
            # Check torque sensor
            current_torque = self.sensor_interface.read_torque_sensor()
            if current_torque < 0.2:  # Minimum holding torque
                logger.critical("GRIP FAILURE - Insufficient torque!")
                self.is_gripping = False
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Grip monitoring error: {e}")
            return False

class KinematicsEngine:
    """4-DOF robot kinematics with real joint limits"""
    
    def __init__(self):
        # Robot arm parameters (mm) - adjust based on your actual robot
        self.link_lengths = {
            'base_height': 150,    # Base to shoulder joint
            'upper_arm': 320,      # Shoulder to elbow
            'forearm': 280,        # Elbow to wrist
            'end_effector': 100    # Wrist to gripper tip
        }
        
        # Joint limits (degrees) - matching Delta servo capabilities
        self.joint_limits = {
            'base': (-180, 180),
            'shoulder': (-90, 90),
            'elbow': (-135, 135),
            'wrist': (-180, 180)
        }
        
        # Workspace limits (mm)
        self.workspace_radius_max = self.link_lengths['upper_arm'] + self.link_lengths['forearm']
        self.workspace_radius_min = abs(self.link_lengths['upper_arm'] - self.link_lengths['forearm'])
        self.workspace_height_max = self.link_lengths['base_height'] + self.workspace_radius_max
        self.workspace_height_min = self.link_lengths['base_height'] - self.workspace_radius_max
    
    def inverse_kinematics(self, target: Position3D, wrist_angle: float = 0) -> Optional[JointAngles]:
        """Calculate joint angles for target position using analytical IK"""
        try:
            x, y, z = target.x, target.y, target.z
            
            # Base rotation (simple rotation around Z-axis)
            base_angle = math.degrees(math.atan2(y, x))
            
            # Distance from base axis
            r = math.sqrt(x*x + y*y)
            
            # Adjust target for end effector offset
            r_eff = r - self.link_lengths['end_effector'] * math.cos(math.radians(wrist_angle))
            z_eff = z - self.link_lengths['base_height'] - self.link_lengths['end_effector'] * math.sin(math.radians(wrist_angle))
            
            # Check workspace reachability
            target_distance = math.sqrt(r_eff*r_eff + z_eff*z_eff)
            
            l1, l2 = self.link_lengths['upper_arm'], self.link_lengths['forearm']
            
            if target_distance > (l1 + l2) * 0.99:  # Add safety margin
                logger.warning(f"Target beyond reach: {target_distance:.1f} > {l1+l2:.1f}")
                return None
                
            if target_distance < abs(l1 - l2) * 1.01:  # Add safety margin
                logger.warning(f"Target too close: {target_distance:.1f} < {abs(l1-l2):.1f}")
                return None
            
            # Elbow angle using law of cosines
            cos_elbow = (l1*l1 + l2*l2 - target_distance*target_distance) / (2*l1*l2)
            cos_elbow = max(-0.999, min(0.999, cos_elbow))  # Clamp to avoid math domain error
            
            # Choose elbow-up solution
            elbow_angle = math.degrees(math.acos(cos_elbow))
            
            # Shoulder angle
            alpha = math.degrees(math.atan2(z_eff, r_eff))
            cos_beta = (l1*l1 + target_distance*target_distance - l2*l2) / (2*l1*target_distance)
            cos_beta = max(-0.999, min(0.999, cos_beta))
            beta = math.degrees(math.acos(cos_beta))
            
            shoulder_angle = alpha - beta
            
            # Wrist angle to maintain desired orientation
            calculated_wrist_angle = wrist_angle - (shoulder_angle + elbow_angle)
            
            # Create joint angles
            angles = JointAngles(base_angle, shoulder_angle, elbow_angle, calculated_wrist_angle)
            
            # Validate joint limits
            if self._validate_joint_limits(angles):
                return angles
            else:
                logger.warning("Calculated angles exceed joint limits")
                return None
                
        except Exception as e:
            logger.error(f"Inverse kinematics error: {e}")
            return None
    
    def forward_kinematics(self, angles: JointAngles) -> Position3D:
        """Calculate end effector position from joint angles"""
        try:
            # Convert to radians
            theta1 = math.radians(angles.base)
            theta2 = math.radians(angles.shoulder)
            theta3 = math.radians(angles.elbow)
            theta4 = math.radians(angles.wrist)
            
            # Link lengths
            l0 = self.link_lengths['base_height']
            l1 = self.link_lengths['upper_arm']
            l2 = self.link_lengths['forearm']
            l3 = self.link_lengths['end_effector']
            
            # Forward kinematics equations
            # Position of elbow joint
            x1 = l1 * math.cos(theta2)
            z1 = l1 * math.sin(theta2)
            
            # Position of wrist joint
            x2 = x1 + l2 * math.cos(theta2 + theta3)
            z2 = z1 + l2 * math.sin(theta2 + theta3)
            
            # Position of end effector
            x3 = x2 + l3 * math.cos(theta2 + theta3 + theta4)
            z3 = z2 + l3 * math.sin(theta2 + theta3 + theta4)
            
            # Convert to world coordinates
            x_world = x3 * math.cos(theta1)
            y_world = x3 * math.sin(theta1)
            z_world = z3 + l0
            
            return Position3D(x_world, y_world, z_world)
            
        except Exception as e:
            logger.error(f"Forward kinematics error: {e}")
            return Position3D(0, 0, 0)
    
    def _validate_joint_limits(self, angles: JointAngles) -> bool:
        """Validate joint angles against limits"""
        joint_values = [
            (angles.base, 'base'),
            (angles.shoulder, 'shoulder'),
            (angles.elbow, 'elbow'),
            (angles.wrist, 'wrist')
        ]
        
        for angle, joint_name in joint_values:
            min_limit, max_limit = self.joint_limits[joint_name]
            if angle < min_limit or angle > max_limit:
                logger.warning(f"Joint {joint_name}: {angle:.1f}° exceeds limits [{min_limit}, {max_limit}]")
                return False
        
        return True
    
    def is_position_reachable(self, target: Position3D) -> bool:
        """Check if target position is within robot workspace"""
        # Check height limits
        if target.z < self.workspace_height_min or target.z > self.workspace_height_max:
            return False
        
        # Check radial distance
        r = math.sqrt(target.x*target.x + target.y*target.y)
        if r < self.workspace_radius_min or r > self.workspace_radius_max:
            return False
        
        # Try inverse kinematics to confirm
        angles = self.inverse_kinematics(target)
        return angles is not None

class SmartGridSystem:
    """Dynamic grid system for path planning"""
    
    def __init__(self, workspace_size: Tuple[float, float, float], resolution: float = 25.0):
        self.workspace_size = workspace_size
        self.resolution = resolution
        
        self.grid_width = int(workspace_size[0] / resolution)
        self.grid_height = int(workspace_size[1] / resolution)
        self.grid_depth = int(workspace_size[2] / resolution)
        
        # Grid maps
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.safety_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        logger.info(f"Grid system initialized: {self.grid_width}x{self.grid_height} at {resolution}mm resolution")
    
    def world_to_grid(self, pos: Position3D) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        x = int((pos.x + self.workspace_size[0]/2) / self.resolution)
        y = int((pos.y + self.workspace_size[1]/2) / self.resolution)
        
        x = max(0, min(self.grid_width - 1, x))
        y = max(0, min(self.grid_height - 1, y))
        
        return (x, y)
    
    def grid_to_world(self, grid_pos: Tuple[int, int], z: float = 0) -> Position3D:
        """Convert grid indices to world coordinates"""
        x = (grid_pos[0] * self.resolution) - self.workspace_size[0]/2
        y = (grid_pos[1] * self.resolution) - self.workspace_size[1]/2
        return Position3D(x, y, z)
    
    def update_obstacles(self, objects: List[DetectedObject]):
        """Update obstacle and safety maps"""
        self.obstacle_map.fill(0)
        self.safety_map.fill(0)
        
        for obj in objects:
            grid_pos = self.world_to_grid(obj.position)
            
            # Calculate object footprint
            obj_width_cells = max(1, int(obj.dimensions[0] / self.resolution))
            obj_height_cells = max(1, int(obj.dimensions[1] / self.resolution))
            
            # Mark obstacle area
            x_start = max(0, grid_pos[0] - obj_width_cells//2)
            x_end = min(self.grid_width, grid_pos[0] + obj_width_cells//2 + 1)
            y_start = max(0, grid_pos[1] - obj_height_cells//2)
            y_end = min(self.grid_height, grid_pos[1] + obj_height_cells//2 + 1)
            
            self.obstacle_map[y_start:y_end, x_start:x_end] = 255
            
            # Safety zone
            safety_margin = 2
            safety_x_start = max(0, x_start - safety_margin)
            safety_x_end = min(self.grid_width, x_end + safety_margin)
            safety_y_start = max(0, y_start - safety_margin)
            safety_y_end = min(self.grid_height, y_end + safety_margin)
            
            self.safety_map[safety_y_start:safety_y_end, safety_x_start:safety_x_end] = 128
    
    def is_cell_free(self, grid_pos: Tuple[int, int], include_safety: bool = True) -> bool:
        """Check if grid cell is free for navigation"""
        x, y = grid_pos
        
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False
        
        if self.obstacle_map[y, x] > 0:
            return False
        
        if include_safety and self.safety_map[y, x] > 0:
            return False
        
        return True

class PathPlanner:
    """A* path planning with grid-based navigation"""
    
    def __init__(self, grid_system: SmartGridSystem):
        self.grid_system = grid_system
    
    def plan_path(self, start: Position3D, goal: Position3D, 
                  avoid_objects: List[DetectedObject]) -> List[Position3D]:
        """Plan optimal path using A* algorithm"""
        try:
            self.grid_system.update_obstacles(avoid_objects)
            
            start_grid = self.grid_system.world_to_grid(start)
            goal_grid = self.grid_system.world_to_grid(goal)
            
            path_grid = self._a_star_search(start_grid, goal_grid)
            
            if not path_grid:
                logger.warning("No path found")
                return [start, goal]  # Direct path as fallback
            
            # Convert to world coordinates
            path_world = []
            for grid_pos in path_grid:
                world_pos = self.grid_system.grid_to_world(grid_pos, start.z)
                path_world.append(world_pos)
            
            # Interpolate Z coordinates
            for i, point in enumerate(path_world):
                progress = i / (len(path_world) - 1) if len(path_world) > 1 else 0
                point.z = start.z + (goal.z - start.z) * progress
            
            logger.info(f"Planned path with {len(path_world)} waypoints")
            return path_world
            
        except Exception as e:
            logger.error(f"Path planning error: {e}")
            return [start, goal]
    
    def _a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm"""
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def get_neighbors(pos):
            neighbors = []
            x, y = pos
            
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.grid_system.grid_width and 
                    0 <= ny < self.grid_system.grid_height):
                    
                    if self.grid_system.is_cell_free((nx, ny)):
                        cost = math.sqrt(dx*dx + dy*dy)
                        neighbors.append(((nx, ny), cost))
            
            return neighbors
        
        # A* implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        visited = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor, move_cost in get_neighbors(current):
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []

class RobotController:
    """Main robot controller integrating all systems"""
    
    def __init__(self, workspace_size: Tuple[float, float, float] = (1000, 1000, 500)):
        self.state = RobotState.IDLE
        self.workspace_size = workspace_size
        
        # Initialize hardware systems
        try:
            self.servo_controller = DeltaServoController()
            self.vision_system = RealSenseVisionSystem()
            self.sensor_interface = SensorInterface()
            self.gripper = VacuumGripperSystem(self.sensor_interface)
            
            # Initialize software systems
            self.kinematics = KinematicsEngine()
            self.grid_system = SmartGridSystem(workspace_size)
            self.path_planner = PathPlanner(self.grid_system)
            
            # Robot state
            self.current_position = Position3D(0, 300, 200)  # Safe home position
            self.current_angles = JointAngles(0, -30, 60, -30)
            self.detected_objects = []
            
            # Performance tracking
            self.stats = {
                'successful_operations': 0,
                'failed_operations': 0,
                'total_time': 0.0
            }
            
            logger.info("Robot controller initialized successfully")
            
        except Exception as e:
            logger.error(f"Robot initialization error: {e}")
            self.state = RobotState.ERROR
    
    def start_system(self) -> bool:
        """Initialize and start robot system"""
        try:
            logger.info("Starting robot system...")
            
            # Enable servos
            self.servo_controller.enable_servos(True)
            
            # Move to home position
            home_angles = JointAngles(0, -30, 60, -30)
            if not self.servo_controller.move_all_joints(home_angles):
                logger.error("Failed to reach home position")
                return False
            
            self.current_angles = home_angles
            self.current_position = self.kinematics.forward_kinematics(home_angles)
            self.state = RobotState.IDLE
            
            logger.info("Robot system ready")
            return True
            
        except Exception as e:
            logger.error(f"System startup error: {e}")
            return False
    
    def execute_pick_and_place(self, pickup_pos: Position3D, place_pos: Position3D) -> bool:
        """Execute complete pick and place operation"""
        start_time = time.time()
        
        try:
            logger.info(f"Pick and place: {pickup_pos} -> {place_pos}")
            
            # Step 1: Scan for objects
            self.state = RobotState.SCANNING
            self.detected_objects = self.vision_system.detect_objects_3d()
            
            # Step 2: Plan path to pickup approach position
            self.state = RobotState.PLANNING
            pickup_approach = Position3D(pickup_pos.x, pickup_pos.y, pickup_pos.z + 80)
            
            if not self.kinematics.is_position_reachable(pickup_approach):
                logger.error("Pickup approach position unreachable")
                return False
            
            pickup_path = self.path_planner.plan_path(
                self.current_position, pickup_approach, self.detected_objects
            )
            
            # Step 3: Execute path to pickup approach
            self.state = RobotState.MOVING
            if not self._execute_path(pickup_path):
                logger.error("Failed to reach pickup approach")
                return False
            
            # Step 4: Lower to pickup position
            if not self._move_to_position(pickup_pos):
                logger.error("Failed to reach pickup position")
                return False
            
            # Step 5: Pick object
            self.state = RobotState.PICKING
            if not self.gripper.attempt_grip():
                logger.error("Failed to pick object")
                return False
            
            # Step 6: Lift object
            lift_pos = Position3D(pickup_pos.x, pickup_pos.y, pickup_pos.z + 100)
            if not self._move_to_position(lift_pos):
                logger.error("Failed to lift object")
                return False
            
            # Step 7: Plan path to place position
            self.state = RobotState.PLANNING
            place_approach = Position3D(place_pos.x, place_pos.y, place_pos.z + 80)
            
            if not self.kinematics.is_position_reachable(place_approach):
                logger.error("Place approach position unreachable")
                return False
            
            place_path = self.path_planner.plan_path(
                lift_pos, place_approach, self.detected_objects
            )
            
            # Step 8: Execute path to place approach
            self.state = RobotState.MOVING
            if not self._execute_path(place_path):
                logger.error("Failed to reach place approach")
                return False
            
            # Step 9: Lower to place position
            if not self._move_to_position(place_pos):
                logger.error("Failed to reach place position")
                return False
            
            # Step 10: Place object
            self.state = RobotState.PLACING
            if not self.gripper.release_grip():
                logger.error("Failed to place object")
                return False
            
            # Step 11: Retract to safe position
            retract_pos = Position3D(place_pos.x, place_pos.y, place_pos.z + 100)
            self._move_to_position(retract_pos)
            
            # Operation successful
            operation_time = time.time() - start_time
            self.stats['successful_operations'] += 1
            self.stats['total_time'] += operation_time
            
            self.state = RobotState.IDLE
            logger.info(f"Pick and place completed in {operation_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Pick and place failed: {e}")
            self.stats['failed_operations'] += 1
            self.state = RobotState.ERROR
            return False
    
    def _move_to_position(self, target: Position3D) -> bool:
        """Move robot to target position"""
        try:
            # Calculate inverse kinematics
            target_angles = self.kinematics.inverse_kinematics(target)
            
            if not target_angles:
                logger.error(f"Cannot reach position: {target}")
                return False
            
            # Execute motion
            if not self.servo_controller.move_all_joints(target_angles):
                logger.error("Servo motion failed")
                return False
            
            # Update current state
            self.current_angles = target_angles
            self.current_position = self.kinematics.forward_kinematics(target_angles)
            
            return True
            
        except Exception as e:
            logger.error(f"Move to position error: {e}")
            return False
    
    def _execute_path(self, path: List[Position3D]) -> bool:
        """Execute planned path"""
        try:
            for i, waypoint in enumerate(path):
                logger.info(f"Moving to waypoint {i+1}/{len(path)}")
                
                if not self._move_to_position(waypoint):
                    logger.error(f"Failed to reach waypoint {i+1}")
                    return False
                
                # Check grip integrity during motion
                if not self.gripper.monitor_grip_integrity():
                    logger.error("Grip failure during path execution")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Path execution error: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop all systems"""
        try:
            logger.critical("EMERGENCY STOP ACTIVATED")
            self.servo_controller.emergency_stop()
            self.gripper.deactivate_vacuum()
            self.state = RobotState.EMERGENCY_STOP
        except Exception as e:
            logger.error(f"Emergency stop error: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'state': self.state.value,
            'position': {
                'x': self.current_position.x,
                'y': self.current_position.y,
                'z': self.current_position.z
            },
            'joint_angles': {
                'base': self.current_angles.base,
                'shoulder': self.current_angles.shoulder,
                'elbow': self.current_angles.elbow,
                'wrist': self.current_angles.wrist
            },
            'gripper': {
                'is_gripping': self.gripper.is_gripping,
                'vacuum_active': self.gripper.vacuum_active,
                'grip_pressure': self.gripper.grip_pressure
            },
            'sensors': {
                'proximity': self.sensor_interface.read_proximity_sensor(),
                'torque': self.sensor_interface.read_torque_sensor()
            },
            'detected_objects': len(self.detected_objects),
            'performance': self.stats
        }
    
    def shutdown_system(self):
        """Safely shutdown robot system"""
        try:
            logger.info("Shutting down robot system...")
            
            # Disable servos
            self.servo_controller.enable_servos(False)
            
            # Turn off gripper
            self.gripper.deactivate_vacuum()
            
            # Stop camera
            self.vision_system.stop_streaming()
            
            # Cleanup GPIO
            self.sensor_interface.cleanup()
            
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

def run_hardware_demo():
    """Run hardware demonstration"""
    robot = None
    
    try:
        # Initialize robot
        robot = RobotController()
        
        if not robot.start_system():
            logger.error("Failed to start robot system")
            return
        
        # Test positions (adjust based on your workspace)
        test_operations = [
            (Position3D(400, 100, 100), Position3D(300, 250, 100)),
            (Position3D(-200, 200, 45), Position3D(-350, -250, 55)),
            (Position3D(150, -180, 40), Position3D(300, -300, 50))
        ]
        
        logger.info(f"Starting {len(test_operations)} pick and place operations")
        
        for i, (pickup, place) in enumerate(test_operations):
            logger.info(f"\n=== Operation {i+1}/{len(test_operations)} ===")
            
            success = robot.execute_pick_and_place(pickup, place)
            
            if success:
                logger.info(f"Operation {i+1} completed successfully")
            else:
                logger.error(f"Operation {i+1} failed")
                break
            
            # Status report
            status = robot.get_system_status()
            logger.info(f"System status: {json.dumps(status, indent=2)}")
            
            time.sleep(2)  # Brief pause between operations
        
        # Final performance report
        final_stats = robot.get_system_status()['performance']
        total_ops = final_stats['successful_operations'] + final_stats['failed_operations']
        success_rate = (final_stats['successful_operations'] / total_ops * 100) if total_ops > 0 else 0
        avg_time = (final_stats['total_time'] / final_stats['successful_operations']) if final_stats['successful_operations'] > 0 else 0
        
        logger.info(f"\n=== PERFORMANCE REPORT ===")
        logger.info(f"Total operations: {total_ops}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Average time per operation: {avg_time:.2f}s")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
    finally:
        if robot:
            robot.shutdown_system()

if __name__ == "__main__":
    # Hardware safety check
    if not HARDWARE_AVAILABLE:
        logger.warning("Running in simulation mode - hardware libraries not available")
        logger.warning("Install required packages: pip install pyrealsense2 pymodbus Jetson.GPIO")
    
    # Run demonstration
    run_hardware_demo()
    
    print("\n" + "="*60)
    print("Hardware-Integrated 4-DOF Robot Control System")
    print("="*60)
    print("Features:")
    print("• Delta ASD-A2 servo driver integration")
    print("• Intel RealSense D435i 3D vision")
    print("• Vacuum gripper with sensor feedback")
    print("• Real-time path planning and collision avoidance")
    print("• Emergency stop and safety monitoring")
    print("• Performance tracking and diagnostics")

    print("="*60)
