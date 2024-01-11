# Copyright (c) 2023, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Implementation of fully connected network."""

from typing import Any, Dict, Union, Iterable
import tensorflow as tf


class FeedForwardNetwork(tf.keras.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size: int, filter_size: int, relu_dropout: float):
    """Initialize FeedForwardNetwork.

    Args:
      hidden_size: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
    super().__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

  def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
    self.filter_dense_layer = tf.keras.layers.Dense(
        self.filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name="filter_layer",
    )
    self.output_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=True, name="output_layer"
    )
    super().build(input_shape)

  def get_config(self) -> Dict[str, Any]:
    return {
        "hidden_size": self.hidden_size,
        "filter_size": self.filter_size,
        "relu_dropout": self.relu_dropout,
    }

  def call(self, x: tf.Tensor, training: bool) -> Dict[str, tf.Tensor]:
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      Dictionary with the following (key:value) pairs:
        "main_output": Output of the feedforward network with shape [batch_size,
        length, hidden_size]. Used as input to the next encoder layer.
    """
    # Retrieve dynamically known shapes

    output = self.filter_dense_layer(x)
    if training:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = self.output_dense_layer(output)
    return dict(main_output=output)
