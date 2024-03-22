from collections import defaultdict
from operator import itemgetter


class mathdict(defaultdict):  # noqa

    @classmethod
    def counter(cls, collection):
        from collections import Counter
        return cls(int, Counter(collection))

    def __header(self, other):
        if isinstance(other, self.__class__):
            raise NotImplementedError
        if not isinstance(other, (int, float)):
            raise TypeError
        new = self.copy()
        new.clear()
        return new

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.default_factory(), (int, float)):
            raise TypeError

    def __repr__(self):
        return f"mathdict({type(self.default_factory())}, {dict(self)})"

    def __neg__(self):
        return self.neg(inplace=False)

    def neg(self, inplace=False):
        mathdict_ = self if inplace else self.copy()
        for k in mathdict_.keys():
            mathdict_[k] = -mathdict_[k]
        return mathdict_

    @staticmethod
    def __add(x, y):
        return x + y

    def __add__(self, other):
        return self.add(other, inplace=False)

    def __radd__(self, other):
        return self.__add__(other)

    def add(self, other, inplace=False, intersection=False):
        return self.operation(self.__add, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __sub(x, y):
        return x - y

    def __sub__(self, other):
        return self.sub(other, inplace=False)

    def __rsub__(self, other):
        return self.__sub__(other).__neg__()

    def sub(self, other, inplace=False, intersection=False):
        return self.operation(self.__sub, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __mul(x, y):
        return x * y

    def __mul__(self, other):
        return self.mul(other, inplace=False, intersection=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def mul(self, other, inplace=False, intersection=False):
        return self.operation(self.__mul, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __floordiv(x, y):
        return x // y

    def __floordiv__(self, other):
        return self.floordiv(other, inplace=False)

    def floordiv(self, other, inplace=False, intersection=False):
        return self.operation(self.__floordiv, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __rfloordiv(x, y):
        return y // x

    def __rfloordiv__(self, other):
        return self.rfloordiv(other, inplace=False)

    def rfloordiv(self, other, inplace=False, intersection=False):
        return self.operation(self.__rfloordiv, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __truediv(x, y):
        return x / y

    def __truediv__(self, other):
        return self.truediv(other, inplace=False)

    def truediv(self, other, inplace=False, intersection=False):
        return self.operation(self.__truediv, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __rtruediv(x, y):
        return y / x

    def __rtruediv__(self, other):
        return self.rtruediv(other)

    def rtruediv(self, other, inplace=False, intersection=False):
        return self.operation(self.__rtruediv, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __mod(x, y):
        return x % y

    def __mod__(self, other):
        return self.mod(other, inplace=False)

    def mod(self, other, inplace=False, intersection=False):
        return self.operation(self.__mod, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __rmod(x, y):
        return y % x

    def __rmod__(self, other):
        return self.rmod(other, inplace=False)

    def rmod(self, other, inplace=False, intersection=False):
        return self.operation(self.__rmod, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __pow(x, y):
        return x ** y

    def __pow__(self, other):
        return self.pow(other, inplace=False)

    def pow(self, other, inplace=False, intersection=False):
        return self.operation(self.__pow, other, inplace=inplace,
                              intersection=intersection)

    @staticmethod
    def __rpow(x, y):
        return y ** x

    def __rpow__(self, other):
        return self.rpow(other, inplace=False)

    def rpow(self, other, inplace=False, intersection=False):
        return self.operation(self.__rpow, other, inplace=inplace,
                              intersection=intersection)

    def operation(self, f, other, inplace=False, intersection=False):
        if intersection and isinstance(other, (dict, self.__class__)):
            new = mathdict(float, {})
            for k in set(self.keys()).intersection(other.keys()):
                new[k] = f(self[k], other[k])
            return new
        else:
            mathdict_ = self if inplace else self.copy()
            if isinstance(other, (int, float)):
                for k in mathdict_.keys():
                    mathdict_[k] = f(mathdict_[k], other)
            elif isinstance(other, (dict, self.__class__)):
                for k in set(mathdict_.keys()).union(other.keys()):
                    try:
                        mathdict_[k] = f(mathdict_[k], other.get(k, 0))
                    except ZeroDivisionError:
                        raise ZeroDivisionError(f'Missing key "{k}"')
            else:
                raise NotImplementedError(f"Does not support {type(other)}")
            return mathdict_

    def __eq__(self, other):
        return bool(self.eq(other))

    def __ge__(self, other):
        return bool(self.ge(other))

    def __gt__(self, other):
        return bool(self.gt(other))

    def __le__(self, other):
        return bool(self.le(other))

    def __lt__(self, other):
        return bool(self.lt(other))

    def __ne__(self, other):
        return bool(self.ne(other))

    def inner(self, other):
        return sum(self.mul(other, intersection=True).values())

    def __add_keys(self, other):
        for k in other.keys():
            _ = self[k]
        return self.keys()

    def eq(self, other):
        new = self.__header(other)
        for k, v in self.items():
            if v == other:
                new[k] = v
        return new

    def ge(self, other):
        new = self.__header(other)
        for k, v in self.items():
            if v >= other:
                new[k] = v
        return new

    def gt(self, other):
        new = self.__header(other)
        for k, v in self.items():
            if v > other:
                new[k] = v
        return new

    def le(self, other):
        new = self.__header(other)
        for k, v in self.items():
            if v <= other:
                new[k] = v
        return new

    def lt(self, other):
        new = self.__header(other)
        for k, v in self.items():
            if v < other:
                new[k] = v
        return new

    def ne(self, other):
        new = self.__header(other)
        for k, v in self.items():
            if v == other:
                new[k] = v
        return new

    def sorted(self, reverse=False):
        return sorted(self.items(), key=itemgetter(1), reverse=reverse)

    def head(self, n=10):
        return self.sorted(reverse=True)[:n]

    def tail(self, n=10):
        return self.sorted(reverse=True)[-n:]

    def sum(self):
        return sum(self.values())

    def freq(self, ignore_zero_division_error=False, inplace=False):
        mathdict_ = self if inplace else self.copy()
        sum = mathdict_.sum()
        if sum == 0 and ignore_zero_division_error:
            return mathdict_
        mathdict_.default_factory = float
        return mathdict_.truediv(sum, inplace=True)

    def to_lists(self):
        return list(self.keys()), list(self.values())

    def filter(self, fn, inplace=False):
        mathdict_ = self if inplace else self.copy()
        for k, v in mathdict_.copy().items():
            if not fn(k, v):
                del mathdict_[k]
        return mathdict_

    def filter_keys(self, fn, inplace=False):
        return self.filter(lambda k, _: fn(k), inplace=inplace)

    def filter_values(self, fn, inplace=False):
        return self.filter(lambda _, v: fn(v), inplace=inplace)

    def truncate(self, n=10):
        return mathdict(self.default_factory, dict(self.head(n)))
