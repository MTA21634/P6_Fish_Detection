import numpy as np


class moving_average_subtractor:
    def __init__(self, a, T):
        self.model = None
        self.a = a
        self.T = T

    def apply(self, img):
        if self.model is not None:
            (height, width) = img.shape
            result = np.zeros((height, width), np.uint8)

            for i in range(height):
                for j in range(width):
                    diff = max(self.model[i, j], img[i, j]) - min(self.model[i, j], img[i, j])

                    if diff > self.T:
                        result[i, j] = 255
                    else:
                        result[i, j] = 0

            self.update_model(img)

            return result
        else:
            self.model = img
            return img

    def update_model(self, img):
        (height, width) = img.shape
        result_model = np.zeros((height, width), np.uint8)

        for i in range(height):
            for j in range(width):
                result_model[i, j] = (1 - self.a) * self.model[i, j] + self.a * img[i, j]

        self.model = result_model
