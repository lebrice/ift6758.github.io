import tensorflow as tf
import warnings
import contextlib
import socket

from typing import *
hostnames = ["fabrice", "marie", "isa", "remi"] 
machine_hostname = socket.gethostname()
# DEBUG should be True if running scripts locally.
# This will automatically use the debug_data, and print more information to the console.
DEBUG = any(name in machine_hostname for name in hostnames)

class PrintLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def call(self, inputs):
        tf.print(inputs)
        return inputs

@contextlib.contextmanager
def log_to_file(path: str):
    # redirect print output to a log file if not in DEBUG mode
    if DEBUG:
        yield
    else:
        with open(path, "a") as f:
            with contextlib.redirect_stdout(f):
                yield

def random_multihot_vector(num_examples, num_classes, prob_1: float = 0.5) -> tf.Tensor:
    """Creates a multi-hot random 'likes' vector.
    
    Keyword Arguments:
        prob_1 {float} -- the probability of having a '1' at each entry. (default: {0.5})
    
    Returns:
        tf.Tensor -- a multi-hot vector of shape [num_examples, num_like_pages], and of dtype tf.bool
    """
    return tf.cast(tf.random.categorical(
        logits=tf.math.log([[1 - prob_1, prob_1] for _ in range(num_examples)]),
        num_samples=num_classes,
        dtype=tf.int32,
    ), tf.bool)


class EarlyStoppingWhenValueExplodes(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', max_value=1e5, verbose = True, check_every_batch=False):
        super().__init__()
        self.monitor = monitor
        self.max_value = max_value
        self.verbose = verbose
        self.check_every_batch = check_every_batch

    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        if self.check_every_batch:
            self.check(batch, logs)
         
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        if not self.check_every_batch:
            self.check(epoch, logs)

    def check(self, t: int, logs: Dict[str, Any]):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(RuntimeWarning(f"Early stopping requires {self.monitor} available!"))

        elif current > self.max_value:
            if self.verbose:
                print(f"\n\n{'Batch' if self.check_every_batch else 'Epoch'} {t}: Early stopping because loss is greater than max value ({self.monitor} = {current})\n\n")
            self.model.stop_training = True

def flatten_dict(nested_dict: Dict[str, Union[Dict, Any]]) -> Dict[str, Any]:
    flattened = {}
    for key, value in nested_dict.items():
        if not isinstance(value, dict):
            flattened[key] = value
        else:
            flattened_child = flatten_dict(value)
            for child_key, child_value in flattened_child.items():
                flattened[key + "." + child_key] = child_value
    return flattened
    