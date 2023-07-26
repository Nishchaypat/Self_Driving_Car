import cv2

from jetson.utils import videoSource, videoOutput, cudaToNumpy, cudaDeviceSynchronize
import numpy as np


class LaneDetection:
    def __init__(self, stepy: int = 50, x1: int = 0, x2: int = 1280):
        self.stepy = stepy
        self.x1 = x1
        self.x2 = x2

    def correct_dist(self, initial_img):
        k = [
            [1.15422732e03, 0.00000000e00, 6.71627794e02],
            [0.00000000e00, 1.14818221e03, 3.86046312e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
        k = np.array(k)
        # Distortion Matrix
        dist = [
            [
                -2.42565104e-01,
                -4.77893070e-02,
                -1.31388084e-03,
                -8.79107779e-05,
                2.20573263e-02,
            ]
        ]
        dist = np.array(dist)
        img_2 = cv2.undistort(initial_img, k, dist, None, k)

        return img_2

    def bgr2rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_roi_image(self, img):
        """
        Our roi is the bottom of the image.
        """
        h, w, channels = img.shape
        # this is horizontal division
        half2 = h // 2
        return img[half2:, :]

    def find_green_lanes(self, img):
        """
        Get the green lanes from the image
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imwrite("test_image/hsv.jpg", hsv)
        ## mask of green (36,25,25) ~ (86, 255,255)
        mask = cv2.inRange(hsv, (36, 25, 25), (94, 255, 255))
        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        return green

    def rgb_to_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def get_x_value_for_given_y(self, img):
        """
        Get the x pixel value for the given y pixel where the lanes are green
        """
        y = range(0, img.shape[0], self.stepy)
        final_list = []
        final_dict = {}
        for y_index in y:
            horizontal_line = img[y_index, self.x1 : self.x2]
            list1 = []
            for i in range(0, horizontal_line.shape[0], 15):
                my_list = horizontal_line[i : i + 15]
                if all(c > 0 for c in my_list):
                    list1.append(i)
            final_dict[y_index] = list1
            final_list.append(list1)
        result_dict = {
            key: value
            for key, value in final_dict.items()
            if isinstance(value, list) and len(value) > 0 and value[-1] - value[0] > 200
        }
        return result_dict

    def get_x_value_for_given_lanes(self, lane_dict):
        points = {}
        for point_y in lane_dict.keys():
            point_list = lane_dict[point_y]
            if len(point_list) > 0:
                points_tuple = (self.x1 + point_list[0], self.x1 + point_list[-1])
                points[point_y] = points_tuple
        return points

    def get_midpoint_for_given_lanes(self, lane_dict):
        first_key = next(iter(lane_dict))
        arr = np.array(lane_dict[first_key])
        average = np.sum(arr) / len(lane_dict[first_key])
        x, y = average, first_key
        return x, y

    def lane_deviation(self, image):
        x_midpoint = (self.x2 - self.x1) / 2.0
        image = self.correct_dist(image)
        image = self.get_roi_image(image)

        image = self.find_green_lanes(image)
        image = self.rgb_to_grayscale(image)

        xy_dict = self.get_x_value_for_given_y(image)
        xy_dict = self.get_x_value_for_given_lanes(xy_dict)

        try:
            x_target, y = self.get_midpoint_for_given_lanes(xy_dict)
            return x_target - x_midpoint
        except StopIteration:
            return None