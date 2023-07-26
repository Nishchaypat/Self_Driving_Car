import cv2
from jetson.utils import videoSource, videoOutput, cudaToNumpy, cudaDeviceSynchronize


from time import perf_counter, sleep

from sign_detection import SignDetection
from lane_detection import LaneDetection
from serial_car_control import SerialController


def start_loop(
    camera,
    sign_detection: SignDetection,
    lane_detection: LaneDetection,
    sc: SerialController,
):
    if camera is None:
        raise RuntimeError("Camera must be defined when starting.")
    if sign_detection is None:
        raise RuntimeError("sign_detection must be defined when starting.")
    if lane_detection is None:
        raise RuntimeError("lane_detection must be defined before starting")

    no_lane_count = 0
    searched_frame = 0
    last_deviation = 0
    img = camera.Capture()
    image_array = cudaToNumpy(img)
    image_array = cv2.rotate(image_array, cv2.ROTATE_180)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = sign_detection.predict_func(image_array)
    try:
        count = 0
        while True:
            start_time = perf_counter()
            count += 1
            img = camera.Capture()
            image_array = cudaToNumpy(img)
            # rotate image 180 degrees
            image_array = cv2.rotate(image_array, cv2.ROTATE_180)
            # change BGR image to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # Need to figure out how often to run and how interrupts will happen
            # SIGN PREDICTION GOES HERE!
            if count % 5 == 0:
                
                boxes, labels, probs = sign_detection.predict_func(image_array)

                x_min, y_min, x_max, y_max  = boxes[max_prob_index]
                x_min = np.round(x_min.item())
                y_min = np.round(y_min.item())
                x_max = np.round(x_max.item())
                y_max = np.round(y_max.item())
            
                
                width, height = (x_max - x_min), (y_max - y_min)
            
                if width> 560 and height > 380:
                    if len(labels) == 0:
                        sign_str = "No Sign."
                    else:
                        if labels[0] == 1:
                            sc.sign_turn_right()
                            sign_str = "RIGHT!"
                        elif labels[0] == 2:
                            sc.sign_turn_left()
                            sign_str = "LEFT!"
                        elif labels[0] == 3:
                            sc.stop()
                            sign_str = "STOP!"
            else:
                sign_str = "SKIP"

            # LANE DETECTION GOES HERE
            deviation = lane_detection.lane_deviation(image_array)
            if deviation is None:
                no_lane_count += 1
                left_motor, right_motor = (0, 0)
                lane_str = "No Lane. count: {}".format(no_lane_count)
            else:
                no_lane_count = 0
                lane_str = str(deviation)
                left_motor, right_motor = sc.lane_correction(deviation)


            if no_lane_count > 5:
                # search stage
                if searched_frame is 0:
                    print("searching left.")
                    sc.turn_left(0.2)
                    searched_frame += 1
                elif searched_frame is 1:
                    print("searching right")
                    sc.turn_right(0.4)
                    searched_frame += 1
                elif searched_frame is 2:
                    print("searching more left")
                    sc.turn_left(0.6)
                    searched_frame +=1 
                elif searched_frame is 3:
                    print("searching more right")
                    sc.turn_right(0.8)
                    searched_frame += 1
                else:
                    sc.turn_left(0.4)
                    print("I'm lost! Can't find a lane!")
                    break
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)
            print(
                    "Sign: {}; Lane dev: {}, {}; L/R: {},{}; fps: {:.2f}".format(
                    sign_str, lane_str, str(last_deviation), left_motor, right_motor, fps
                )
            )
            last_deviation = deviation

        camera.Close()
        sc.close_connection()
    except KeyboardInterrupt:
        camera.Close()
        sc.close_connection()


if __name__ == "__main__":
    camera = videoSource("csi://0")
    sign_detection = SignDetection()
    lane_detection = LaneDetection()
    sc = SerialController("/dev/ttyUSB0", 9600, 0.1, 80)

    cudaDeviceSynchronize()

    start_loop(camera, sign_detection, lane_detection, sc)