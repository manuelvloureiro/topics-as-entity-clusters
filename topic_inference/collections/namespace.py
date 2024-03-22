import json


class Namespace:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        if args:
            self.__dict__.update({"args": args})

    def __contains__(self, key):
        return key in self.__dict__

    def __copy__(self):
        return self.__class__(**self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "<{} ({})>".format(self.__class__.__name__, hex(id(self)))

    def get(self, key, value=None):
        return self.__dict__.get(key, value)

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)
        return self

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.__dict__, sort_keys=True)
