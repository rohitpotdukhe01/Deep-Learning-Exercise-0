import numpy as np
import matplotlib.pyplot as plt


class Checker:
    output = 0

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        white = np.ones((self.resolution, self.resolution), dtype=np.uint8)
        black = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        white[0: self.tile_size, 0: self.tile_size] = black
        white[self.tile_size: self.tile_size * 2, self.tile_size: self.tile_size * 2] = black
        self.output = np.tile(white[0: self.tile_size * 2, 0: self.tile_size * 2],
                              (self.resolution // (self.tile_size * 2), (self.resolution // (self.tile_size * 2))))
        return self.output.copy()

    def show(self):
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    output = 0

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        x_coord = np.arange(0, self.resolution)
        y_coord = np.arange(0, self.resolution)
        x, y = np.meshgrid(x_coord, y_coord)

        circle = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        self.output = circle <= self.radius
        return self.output.copy()

    def show(self):
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    output = 0

    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        self.output = np.zeros([self.resolution, self.resolution, 3])
        self.output[:, :, 0] = np.linspace(0.0, 1.0, self.resolution)
        self.output[:, :, 1] = np.linspace(0.0, 1.0, self.resolution).reshape(self.resolution, 1)
        self.output[:, :, 2] = np.linspace(1.0, 0.0, self.resolution)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()
