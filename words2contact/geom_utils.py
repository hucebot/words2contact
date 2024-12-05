import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def scale(self, scale_factor_width: float, scale_factor_height: float):
        """Scale the point by given factors."""
        self.x *= scale_factor_width
        self.y *= scale_factor_height

    def __add__(self, other):
        """Add a Point or a tuple/list to this Point."""
        if isinstance(other, (tuple, list)):
            return Point(self.x + other[0], self.y + other[1])
        elif isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            raise TypeError(
                f"Cannot add Point with type {type(other)}. Supported types are Point, tuple, and list."
            )

    def sys_prompt(self):
        """Return the system representation of the point."""
        return f"({self.x}, {self.y})"


class BoundingBox:
    def __init__(self, x: float, y: float, width: float, height: float, class_name: str = None):
        """
        Initialize a bounding box.

        Args:
            x: X-coordinate of the bottom-left corner.
            y: Y-coordinate of the bottom-left corner.
            w: Width of the bounding box.
            h: Height of the bounding box.
            class_name: Optional class name for the bounding box.
        """
        self.anchor = Point(x, y)
        self.x = x  # Bottom-left corner X
        self.y = y  # Bottom-left corner Y
        self.w = width  # Width
        self.h = height  # Height
        self.class_name = class_name

    def scale(self, scale_factor_x: float, scale_factor_y: float):
        """Scale the bounding box dimensions and position."""
        self.x *= scale_factor_x
        self.y *= scale_factor_y
        self.w *= scale_factor_x
        self.h *= scale_factor_y

    def __str__(self):
        return (f"BoundingBox(anchor={self.anchor}, width={self.w}, "
                f"height={self.h}, class_name={self.class_name})")

    def plot_bb(self, ax, color='white'):
        """
        Plot the bounding box on the given matplotlib Axes.

        Args:
            ax: Matplotlib Axes object.
            color: Color of the bounding box and text.
        """
        ax.add_patch(plt.Rectangle((self.x, self.y), self.w, self.h,
                                   edgecolor=color, facecolor='none', alpha=0.5))
        # Plot class name at the center of the bounding box if provided
        if self.class_name:
            ax.text(self.x + self.w / 2, self.y + self.h / 2, self.class_name,
                    fontsize=8, color=color, ha='center', va='center')

    def to_int(self):
        """Convert bounding box attributes to integers."""
        self.x = int(self.x)
        self.y = int(self.y)
        self.w = int(self.w)
        self.h = int(self.h)

    def get_sys_prompt(self):
        """
        Return a system-friendly representation of the bounding box.

        Includes the class name, midpoint, and dimensions.
        """
        mid_x = int(self.x + self.w / 2)
        mid_y = int(self.y + self.h / 2)
        return (f"{self.class_name} is at ({mid_x}, {mid_y}), "
                f"with dimensions: {int(self.w)}x{int(self.h)}")
