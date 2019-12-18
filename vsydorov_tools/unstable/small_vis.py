"""
Small visualization with matplotlib and stuff
"""
from matplotlib import pyplot as plt  # type: ignore
import matplotlib.image as mpimg
import numpy as np  # type: ignore
import logging
from contextlib import contextmanager
from io import BytesIO
from typing import Dict, List, Set, Tuple, Union  # NOQA

log = logging.getLogger(__name__)


# Draw with matplotlib

def plt_curve_x(ax, x, y_label, label=None):
    ax.plot(x, label=label)
    ax.minorticks_on()
    ax.set_ylabel(y_label)
    if label:
        ax.legend()


def bytes_to_matplotlib(bytes_):
    bio = BytesIO()
    bio.write(bytes_)
    bio.seek(0)
    img = mpimg.imread(bio)
    return img


def plt_curves_xy(
        ax,
        xys: List[Union[np.ndarray]],
        x_label: str = None,
        y_label: str = None,
        labels: List[str] = None):
    """
    Args:
        xy: [ndaray, ndarray], representing X and Y values
    """
    good_labels: List[str] = [None]*len(xys) if labels is None else labels  # type: ignore
    for xy, label in zip(xys, good_labels):
        ax.plot(*xy, label=label)
        ax.scatter(*xy)
    ax.set_xticks(xy[0], minor=True)
    ax.minorticks_on()
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if labels:
        ax.legend()


@contextmanager
def save_subplots(savefile, *args, **kwargs):
    with plt_backend('Agg'):
        f, ax = plt.subplots(*args, **kwargs)
        yield f, ax
        f.savefig(str(savefile), bbox_inches='tight', pad_inches=0)
        plt.close()


def draw_canvas_to_ndarray(fig) -> np.ndarray:
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    npimg = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    npimg = npimg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return npimg


@contextmanager
def plt_backend(backend_name):
    original = plt.get_backend()
    plt.switch_backend(backend_name)
    yield
    plt.switch_backend(original)


def annotate_with_sidewise_arrows(annotate_me):
    """
    annotation:
        'xy': (x,y),
        'label': string
        OPTIONAL
        'color': string
        'arrow': bool
    """
    MAX_SPACING = 0.09
    if not len(annotate_me):
        return
    spacing_ = min(1/len(annotate_me), MAX_SPACING)
    # Annotate with lesser spacing if too many
    for i, annotation in enumerate(annotate_me):
        xy = annotation['xy']
        label = annotation['label']
        facecolor = annotation.get('color', 'black')
        arrow = annotation.get('arrow', True)
        if arrow:
            arrowprops = dict(ec=facecolor, headwidth=0.3, width=0.1, connectionstyle="arc3,rad=.03")
        else:
            arrowprops = None
        plt.annotate(label,
                xy=xy, xycoords='data',
                xytext=(1, 1-i*spacing_), textcoords='axes fraction',
                arrowprops=arrowprops,
                verticalalignment='bottom',
                fontsize='x-small',
                color=facecolor)


def draw_bbox(bbox, color='red', text=None, width=1, ax=None):
    try:
        _l, _t, _r, _d = bbox.l, bbox.t, bbox.r, bbox.d
    except AttributeError:
        _l, _t, _r, _d = bbox

    if ax is None:
        ax = plt.gca()

    ax.add_patch(plt.Rectangle((_l, _t),
        _r-_l,
        _d-_t,
        alpha=1,
        facecolor='none',
        edgecolor=color,
        linewidth=width))

    if text is not None:
        ax.text(_l, _t, text, color=color)


def plot_p3(p3, color='black', ax=None):
    """
        p3 basis:
        0 (0, 0) ---> 2 (X, 0)
        |
        V
        1 (0, Y)
    """

    def base_arrow(center, vector):
        ax.arrow(*center, *vector,
                head_width=np.mean(vector)/40,
                head_length=np.mean(vector)/20,
                fc=color, ec=color, clip_on=False)

    def striped_arrow(center, vector):
        ax.arrow(*center, *vector,
                linestyle=':', fc=color, ec=color, clip_on=False)

    if ax is None:
        ax = plt.gca()
    vec01 = p3[1] - p3[0]  # 0 -> 1 vector
    vec02 = p3[2] - p3[0]  # 0 -> 1 vector
    base_arrow(p3[0], vec01)  # 0 --> 1 arrow
    base_arrow(p3[0], vec02)  # 0 --> 2 arrow
    striped_arrow(p3[1], vec02)  # 1 --> unnamed point
    striped_arrow(p3[2], vec01)  # 2 --> unnamed point
