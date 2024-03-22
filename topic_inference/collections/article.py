from .namespace import Namespace


class Article(Namespace):

    def __init__(self, ref, title, body, source, *args, **kwargs):
        self.ref = ref
        self.title = title
        self.body = body
        self.source = source
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "{}: (ref: {}, title: {}, source: {})".format(
            self.__class__.__name__,
            self.ref,
            self.title,
            self.source
        )

    def __repr__(self):
        return "<{}>".format(str(self))

    def to_text(self, pattern="{} {}"):
        return pattern.format(self.title, self.body)
