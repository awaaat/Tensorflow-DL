# A utility to help as add a function to a class as a method
import inspect

def add_to_class(Class): #@save
    """
    Decorator to dynamically add a function as a method to a specified class.
    Args:
        Class (type): The class to which the function will be added as a method.
    Returns:
        function: A wrapper function that adds the given function to the specified class.
    Usage:
        @add_to_class(MyClass)
        def my_method(self, arg1, arg2):
            # Method implementation
            pass

    Example:
        class MyClass:
            pass

        @add_to_class(MyClass)
        def greet(self, name):
            return f"Hello, {name}!"

        obj = MyClass()
        print(obj.greet("World"))  # Output: "Hello, World!"
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class HyperParameters:
    """
    A class for managing and saving hyperparameters.
    """
    def save_hyperparameters(self, ignore=[]):
        """
        Save the hyperparameters of the class, excluding specified ones.

        Args:
            ignore (list): A list of hyperparameter names to exclude.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


class ProgressBoard(HyperParameters):
    def __init__(self, xlabel = None,
                ylabel = None, xlim = None, 
                ylim = None,  xscale = 'linear', yscale = 'linear',
                ls=['-', '--', '-.', ':'], 
                colors=['C0', 'C1', 'C2', 'C3'], 
                fig=None, axes=None, figsize=(3.5, 2.5), display=True
                ):
        
        self.save_hyperparameters()

