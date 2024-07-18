import matplotlib.colors as mc
import colorsys


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_color_transparent(color, alpha=0.5):
    return color[0], color[1], color[2], alpha
