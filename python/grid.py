import inspect
import itertools

class Grid:
    def __init__(self, params):
        self.estimators = []
        self.cur = 0
        
        for estimator in params['estimator']:
            inspect_args = inspect.signature(estimator).parameters
            args_name = [arg for arg in params if arg in inspect_args.keys()]
            defaults = [inspect_args[a].default for a in args_name]
            args_choices = [list(set(params[a]) | {d}) if (not params[a] and d != inspect._empty) else params[a] for (a,d) in zip(args_name, defaults)]
            for args_val in itertools.product(*args_choices):
                kwargs = dict(zip(args_name, args_val))
                self.estimators.append((estimator, kwargs))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cur < len(self.estimators):
            estimator, kwargs = self.estimators[self.cur]
            self.cur += 1
            return estimator(**kwargs)
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.estimators)
    
    def get_tuples(self):
        return self.estimators