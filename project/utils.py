import tensorflow as tf

from socket import gethostname
DEBUG = "fabrice" in gethostname()

class PrintLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def call(self, inputs):
        tf.print(inputs)
        return inputs



import contextlib

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
