import tensorflow as tf


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self.metrics = {}

    def __call__(self, tag, value, step):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value : scalar value
            value
        step : int
            training iteration
        """
        if tag not in self.metrics:
            _metric = tf.Summary()
            _metric.value.add(tag=tag, simple_value=None)
            self.metrics[tag] = _metric

        summary = self.metrics[tag]

        summary.value[0].simple_value = value
        self.writer.add_summary(summary, step)