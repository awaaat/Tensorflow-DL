import inspect
import collections
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from IPython import display


def use_svg_display():
    """Switches the default matplotlib display format to SVG for sharper plots in Jupyter notebooks."""
    set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Sets the default figure size for matplotlib plots.

    Args:
        figsize (tuple): A tuple specifying the figure dimensions in inches (width, height).
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Configures the appearance of matplotlib axes.

    Args:
        axes: Matplotlib axes to configure.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        xlim (tuple): Limits for the x-axis (min, max).
        ylim (tuple): Limits for the y-axis (min, max).
        xscale (str): Scale for the x-axis ('linear', 'log', etc.).
        yscale (str): Scale for the y-axis ('linear', 'log', etc.).
        legend (list): List of legend labels.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plots data points with specified configurations.

    Args:
        X (list or tensor): Data for the x-axis. Can be 1D or 2D.
        Y (list or tensor): Data for the y-axis. If None, Y is set to X.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        legend (list): List of legend labels.
        xlim (tuple): Limits for the x-axis (min, max).
        ylim (tuple): Limits for the y-axis (min, max).
        xscale (str): Scale for the x-axis ('linear', 'log', etc.).
        yscale (str): Scale for the y-axis ('linear', 'log', etc.).
        fmts (tuple): Line formats for plotting multiple curves.
        figsize (tuple): Size of the figure (width, height) in inches.
        axes: Matplotlib axes to plot on. If None, the current axes are used.
    """
    def has_one_axis(X):  
        """Checks if the input has one axis."""
        return (hasattr(X, "ndim") and X.ndim == 1 or 
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def add_to_class(Class):  
    """Decorator to dynamically add a function as a method to a specified class.

    Args:
        Class (type): The class to which the function will be added.

    Returns:
        function: A wrapper that attaches the function to the class.

    Example:
        class MyClass:
            pass

        @add_to_class(MyClass)
        def greet(self, name):
            return f"Hello, {name}!"
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


class HyperParameters:
    """A base class for managing and saving hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Saves hyperparameters of the class while ignoring specified ones.

        Args:
            ignore (list): List of parameter names to exclude from saving.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


class ProgressBoard(HyperParameters):
    """A class for managing and displaying progress during training or experimentation.

    Inherits from:
        HyperParameters
    """
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'], fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        """Initializes the ProgressBoard with specified configurations.

        Args:
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            xlim (tuple): Limits for the x-axis.
            ylim (tuple): Limits for the y-axis.
            xscale (str): Scale for the x-axis ('linear', 'log', etc.).
            yscale (str): Scale for the y-axis ('linear', 'log', etc.).
            ls (list): Line styles for plotting.
            colors (list): Colors for different lines.
            fig: Matplotlib figure for plotting.
            axes: Matplotlib axes for plotting.
            figsize (tuple): Size of the figure (width, height) in inches.
            display (bool): Whether to display the plot in Jupyter notebooks.
        """
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        """Plots data points and updates the progress board.

        Args:
            x (float): The x-coordinate of the new data point.
            y (float): The y-coordinate of the new data point.
            label (str): Label for the data series.
            every_n (int): Frequency of updating the displayed plot.
        """
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                       linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim:
            axes.set_xlim(self.xlim)
        if self.ylim:
            axes.set_ylim(self.ylim)
        if not self.xlabel:
            self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)
