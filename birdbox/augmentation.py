from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np


class _RandomGenerator(stateful_random_ops.Generator):

    @property
    def state(self):
        """The internal state of the RNG."""
        state_var = self._state_var
        try:
            _ = getattr(state_var, 'handle')
            return state_var
        except ValueError:
            return state_var.values[0]

    def _create_variable(self, *args, **kwargs):
        # This function does the same thing as the base class's namesake, except
        # that it skips the distribution-strategy check. When we are inside a
        # distribution-strategy scope, variables.Variable will pick a proper
        # variable class (e.g. MirroredVariable).
        return variables.Variable(*args, **kwargs)


def _make_generator(seed=None):
    if seed:
        return _RandomGenerator.from_seed(seed)
    else:
        return _RandomGenerator.from_non_deterministic_state()


class BaseLayer(Layer):
    def __init__(self, seed=None, name=None, **kwargs):
        super(BaseLayer, self).__init__(name=name, **kwargs)
        self.seed = seed
        self._rng = _make_generator(self.seed)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'seed': self.seed, }
        base_config = super(BaseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomColorAddition(BaseLayer):
    def __init__(self, lower, upper, channel=0, **kwargs):
        super(RandomColorAddition, self).__init__(**kwargs)
        self.upper = upper
        self.lower = lower
        self.channel_index = channel

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_channel_added_inputs():

            if inputs.shape[0] is None:
                return inputs

            channel_data = []
            for i_channel in range(3):
                if self.channel_index == i_channel:
                    temp = [inputs[i, :, :, i_channel] + self._rng.uniform((1,), self.lower, self.upper) for i in range(inputs.shape[0])]
                    temp = tf.stack(temp, axis=0)
                else:
                    temp = inputs[:, :, :, i_channel]
                channel_data.append(temp)

            channel_data = tf.clip_by_value(channel_data, 0, 255)
            return tf.stack(channel_data, axis=3)


        output = tf_utils.smart_cond(training, random_channel_added_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output


class RandomHue(BaseLayer):
    def __init__(self, max_delta, **kwargs):
        super(RandomHue, self).__init__(**kwargs)
        self.max_delta = max_delta

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_channel_added_inputs():
            return tf.image.random_hue(inputs, self.max_delta)

        output = tf_utils.smart_cond(training, random_channel_added_inputs, lambda: inputs)
        output.set_shape(inputs.shape)
        return output


class RandomSaturation(BaseLayer):
    def __init__(self, lower=0.9, upper=1.1, **kwargs):
        super(RandomSaturation, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def augment():
            outputs = tf.image.random_saturation(inputs, self.lower, self.upper)
            return tf.clip_by_value(outputs, 0, 255)

        output = tf_utils.smart_cond(training, augment, lambda: inputs)
        output.set_shape(inputs.shape)
        return output


class RandomBrightness(BaseLayer):
    def __init__(self, max_delta, **kwargs):
        super(RandomBrightness, self).__init__(**kwargs)
        self.max_delta = max_delta

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def augment():
            outputs = tf.image.random_brightness(inputs, self.max_delta)
            return tf.clip_by_value(outputs, 0, 255)

        output = tf_utils.smart_cond(training, augment, lambda: inputs)
        output.set_shape(inputs.shape)
        return output


