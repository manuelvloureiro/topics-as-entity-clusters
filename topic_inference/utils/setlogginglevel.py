from functools import wraps


def setlogginglevel(logger, level):
    """Decorator to change the logging level of a function."""
    def _suspendlogging(func):
        @wraps(func)
        def inner(*args, **kwargs):
            previousloglevel = logger.getEffectiveLevel()
            logger.setLevel(level)
            try:
                return func(*args, **kwargs)
            finally:
                logger.setLevel(previousloglevel)

        return inner

    return _suspendlogging

