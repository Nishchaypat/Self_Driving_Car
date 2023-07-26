import serial
import json
from time import sleep
from typing import Union, List


class SerialController:
    def __init__(
        self,
        com_port: str,
        baudrate: int,
        timeout: int,
        default_speed: int,
        deviation_factor: int = 0.18,
        deadzone: int = 20,
    ):
        self.ser = serial.Serial(com_port, baudrate, timeout=timeout)
        if deviation_factor >= 1 or deviation_factor < 0:
            raise Exception("Invalid deviation_factor. Must be between 0 and 1.")
        self.deviation_factor = deviation_factor
        if deadzone > 400 or deadzone < 0:
            raise Exception("Invalid deadzone. Must be between 0 and 400")
        self.deadzone = deadzone
        if default_speed < 0 or default_speed > 255:
            raise Exception("Invalid speed. Must be between 0 and 255")
        self.default_speed = default_speed
        self.ser.readlines()

    def close_connection(self):
        self.stop()
        self.ser.close()

    def stop(self):
        payload = self.__build_payload(100)
        self.ser.write(payload)
        self.ser.readline()

    def lane_correction(self, lane_deviation: float):
        # control the maximum values received
        if lane_deviation > 300:
            lane_deviation = 300
        if lane_deviation < -300:
            lane_deviation = -300

        # if the interval value falls in the deadzone, go forward
        if -self.deadzone <= lane_deviation <= self.deadzone:
            self.forward()
            return self.default_speed, self.default_speed

        # with a deviation_factor of 1/8, the difference in motor speeds will not
        # be greater than 50.
        motor_difference = lane_deviation * self.deviation_factor
        if self.default_speed - abs(motor_difference) < 0:
            motor_difference = self.default_speed
        if motor_difference > 0:
            left = self.default_speed
            right = self.default_speed - abs(motor_difference)

        if motor_difference < 0:
            left = self.default_speed - abs(motor_difference)
            right = self.default_speed
        
        self.set_left_and_right(left, right)

        return left, right

    def get_yaw(self) -> float:
        payload = self.__build_payload(120)
        if self.ser.in_waiting > 0:
            self.ser.readall()
        self.ser.write(payload)
        try:
            return float(self.ser.readline())
        except ValueError:
            return None

    def sign_turn_right(self):
        current_yaw = None
        while current_yaw is None:
            current_yaw = self.get_yaw()
        target_yaw = current_yaw + 90.0
        self.turn_right()
        while True:
            print("{} > {}\n".format(current_yaw, target_yaw))
            if self.get_yaw() > target_yaw:
                self.stop()
                return

    def sign_turn_left(self):
        current_yaw = None
        while current_yaw is None:
            current_yaw = self.get_yaw()
        target_yaw = current_yaw - 90.0
        self.turn_left()
        while True:
            print("{} > {}\n".format(current_yaw, target_yaw))
            if self.get_yaw() < target_yaw:
                self.stop()
                return

    def set_left_and_right(self, left: int, right: int):
        payload = self.__build_payload(4, [right, left])
        self.ser.write(payload)
        self.ser.readline()

    def forward(self, time: Union[float, None] = None, speed: Union[int, None] = None):
        self.__movement(1, time, speed)

    def back(self, time: Union[float, None] = None, speed: Union[int, None] = None):
        self.__movement(2, time, speed)

    def turn_left(
        self, time: Union[float, None] = None, speed: Union[int, None] = None
    ):
        self.__movement(3, time, speed)

    def turn_right(
        self, time: Union[float, None] = None, speed: Union[int, None] = None
    ):
        self.__movement(4, time, speed)

    def left_front(
        self, time: Union[float, None] = None, speed: Union[int, None] = None
    ):
        self.__movement(5, time, speed)

    def rear_left(
        self, time: Union[float, None] = None, speed: Union[int, None] = None
    ):
        self.__movement(6, time, speed)

    def right_front(
        self, time: Union[float, None] = None, speed: Union[int, None] = None
    ):
        self.__movement(7, time, speed)

    def rear_right(self, time: Union[float, int, None], speed: Union[int, None] = None):
        self.__movement(8, time, speed)

    def __movement(
        self,
        direction: int,
        time: Union[float, None] = None,
        speed: Union[int, None] = None,
    ):
        if not speed:
            speed = self.default_speed
        payload = self.__build_payload(102, [direction, speed])
        self.ser.write(payload)
        self.ser.readline()
        if time:
            sleep(time)
            self.stop()

    def __build_payload(
        self, input_type: int, parameters: Union[List[int], None] = None
    ):
        payload = {"N": input_type}
        if parameters:
            for index, parameter in enumerate(parameters, start=1):
                payload["D" + str(index)] = parameter

        return json.dumps(payload).encode("ascii")