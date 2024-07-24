import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def __str__(self):
        return f"Point: {self.x, self.y}"

    def scale(self, scale_factor_width: float, scale_factor_height: float):
        self.x = self.x * scale_factor_width
        self.y = self.y * scale_factor_height

    def __add__(self, other):
        if isinstance(other, tuple) or isinstance(other, list):
            return Point(self.x + other[0], self.y + other[1])
        elif isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            raise ValueError("Cannt add Point with type: ", type(
                other), "Supported types are: Point, tuple, list")

    def sys_prompt(self):
        return f"({self.x}, {self.y})"


class BoundingBox:
    def __init__(self, x, y, w, h, class_name=None):

        # This is bottom left corner
        self.anchor = Point(x, y)
        self.x = x  # anchor x
        self.y = y  # anchor y
        self.w = w  # width
        self.h = h  # height
        self.class_name = class_name

    def get_top_left(self):
        return Point(self.x, self.y + self.h)

    def get_top_right(self):
        return Point(self.x + self.w, self.y + self.h)

    def get_bottom_left(self):
        return Point(self.x, self.y)

    def get_bottom_right(self):
        return Point(self.x + self.w, self.y)

    def get_center(self):
        return Point(self.x + self.w/2, self.y + self.h/2)

    def get_mid_top(self):
        return Point(self.x + self.w/2, self.y + self.h)

    def get_mid_bottom(self):
        return Point(self.x + self.w/2, self.y)

    def get_mid_left(self):
        return Point(self.x, self.y + self.h/2)

    def get_mid_right(self):
        return Point(self.x + self.w, self.y + self.h/2)

    def scale(self, scale_factor_x: float, scale_factor_y: float):

        self.x = self.x * scale_factor_x
        self.y = self.y * scale_factor_y
        self.w = self.w * scale_factor_x
        self.h = self.h * scale_factor_y

    def __str__(self):
        return f"Bounding Box: with anchor: {self.anchor}, width: {self.w}, height: {self.h}"

    def upscale_bb(self, original_img, scaled_img):
        original_height, original_width = original_img.size
        height, width = scaled_img.shape
        # scale the bounding box to the original image
        x = original_height * self.x / width
        y = original_width * self.y / height
        w = original_height * self.w / width
        h = original_width * self.h / height
        self.scaled_bb = x, y, w, h

    def plot_bb(self, ax, color='white'):
        ax.add_patch(plt.Rectangle((self.x, self.y), self.w, self.h,
                     edgecolor=color, facecolor='none', alpha=0.5))
        # plot the class name, in the middle of the bounding box
        ax.text(self.x + self.w/2, self.y + self.h/2, self.class_name,
                fontsize=8, color=color, ha='center', va='center')

    def to_int(self):
        self.x = int(self.x)
        self.y = int(self.y)
        self.w = int(self.w)
        self.h = int(self.h)

    def plot(self, ax, legend=None):
        ax.add_patch(plt.Rectangle((self.x, self.y), self.w, self.h,
                     edgecolor='r', facecolor='none', alpha=0.5))

    def get_sys_prompt(self):
        print("Getting sys prompt")
        mid = [int(self.x + self.w/2), int(self.y + self.h/2)]
        # we print the class name, the mid point, and the boundaries (max x, max y, min x, min y)
        return f"{self.class_name} is at {mid}, with dimensions: {int(self.w)}x{int(self.h)}"
