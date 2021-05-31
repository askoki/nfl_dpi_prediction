import math
import numpy as np
from matplotlib.patches import FancyArrowPatch


def home_has_possession(row):
    if row.possessionTeam == row.homeTeamAbbr:
        return True
    return False


def calculate_team_sitation(row):
    ball_string = 'football'
    if row.team == ball_string:
        return ball_string
    if row.team == 'home' and row.homeHasPossession:
        return 'attacking'
    elif row.team == 'away' and not row.homeHasPossession:
        return 'attacking'
    return 'defending'


def convert_speed_to_marker_size(speed: float) -> int:
    if 0 < speed <= 1.5:
        return 10
    elif 1.5 < speed <= 3:
        return 15
    elif 3 < speed <= 4.5:
        return 20
    elif 4.5 < speed <= 6:
        return 25
    return 30


def arrow(x, y, s, ax, color):
    """
    Function to draw the arrow of the movement
    :param x: position on x-axis
    :param y: position on y-axis
    :param s: speed in yards/s
    :param ax: plot's configuration
    :param color: color of the arrows
    :return: arrows on the specific positions
    """
    # distance between the arrows
    distance = 5
    ind = range(1, len(x), distance)

    # computing of the arrows
    for i in ind:
        ar = FancyArrowPatch(
            (x[i - 1], y[i - 1]), (x[i], y[i]),
            arrowstyle='->',
            mutation_scale=convert_speed_to_marker_size(s[i]),
            color=color,
        )
        ax.add_patch(ar)


def calculate_arrow_xy(x, y, o):
    o = o % 360
    delta = 0.1
    if o == 0:
        y_delta = delta
        x_delta = 0
        return x + x_delta, y + y_delta
    elif o == 90:
        y_delta = 0
        x_delta = delta
        return x + x_delta, y + y_delta
    elif o == 180:
        y_delta = -delta
        x_delta = 0
        print(f'F {y_delta}')
        return x + x_delta, y + y_delta
    elif o == 270:
        y_delta = 0
        x_delta = -delta
        return x + x_delta, y + y_delta

    elif 0 < o < 90:
        y_delta = math.sin(math.radians(90 - o)) * delta
        x_delta = math.sqrt(delta ** 2 - y_delta ** 2)
        return x + x_delta, y + y_delta
    elif 90 < o < 180:
        y_delta = math.sin(math.radians(o - 90)) * delta
        x_delta = math.sqrt(delta ** 2 - y_delta ** 2)
        return x + x_delta, y - y_delta
    elif 180 < o < 270:
        x_delta = math.sin(math.radians(o - 180)) * delta
        y_delta = math.sqrt(delta ** 2 - x_delta ** 2)
        return x - x_delta, y - y_delta
    else:
        y_delta = math.sin(math.radians(o - 270)) * delta
        x_delta = math.sqrt(delta ** 2 - y_delta ** 2)
        return x - x_delta, y + y_delta


def arrow_o(x, y, o, s, ax, color):
    """
    Function to draw the arrow of the movement
    :param x: position on x-axis
    :param y: position on y-axis
    :param o: orientation in degrees 0-360
    :param s: speed in yards/s
    :param ax: plot's configuration
    :param color: color of the arrows
    :return: arrows on the specific positions
    """
    # distance between the arrows
    distance = 3
    ind = range(5, len(x), distance)

    # computing of the arrows
    for i in ind:
        x2, y2 = calculate_arrow_xy(x[i], y[i], o[i])
        ar = FancyArrowPatch(
            (x[i], y[i]), (x2, y2),
            arrowstyle='-|>',
            mutation_scale=convert_speed_to_marker_size(s[i]),
            alpha=0.6,
            color=color,
        )
        ax.add_patch(ar)


def calculate_distance_v4(x1: np.array, y1: np.array, x2: np.array, y2: np.array) -> np.array:
    return np.round(np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)), 2)
