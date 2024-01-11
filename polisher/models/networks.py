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
"""TF2 + tf.keras implementations of networks for polisher."""

import logging
from typing import Any, Dict, Optional, Union

import ml_collections
import tensorflow as tf

from polisher.make_images import encoding
from polisher.models import data_providers
from polisher.models import encoder_stack
from polisher.models import model_configs
from official.nlp.modeling import layers


class ModifiedOnDeviceEmbedding(layers.OnDeviceEmbedding):
  """Subclass of OnDeviceEmbedding, init similar to EmbeddingSharedWeights."""

  def __init__(self, vocab_size, embedding_width, **kwargs):
    # Set initializer and scale_factor to match the original implementation in
    # tensorflow_models/official/legacy/transformer/embedding_layer.py
    super().__init__(
        vocab_size,
        embedding_width,
        initializer=tf.random_normal_initializer(
            mean=0.0, stddev=embedding_width**-0.5
        ),
        scale_factor=embedding_width**0.5,
        **kwargs,
    )

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    # make sure 0 ids match to zero emebeddings.
    embeddings = super().call(inputs)
    mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
    embeddings *= tf.expand_dims(mask, -1)
    return embeddings


class EncoderOnlyTransformer(tf.keras.Model):
  """Modified encoder-only transformer model.

  This implementation is similar to

  tensorflow_models/official/legacy/transformer/transformer.py
  tensorflow_models/official/nlp/modeling/models/seq2seq_transformer.py

  with some simplifications and extensions.

  The main changes are:

  * Removing logic relating to converting tokens to embeddings, since the
  DeepConsensus is already in the form of vectors for each position.

  * Removing the decoder, since we only want to run the encoder.

  * Adding additional layers on top of the encoder for the per-position
  classification task.
  """

  def __init__(
      self,
      params: ml_collections.ConfigDict,
      name: Optional[str] = None,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.params = params
    if self.params.add_pos_encoding:
      self.position_embedding = layers.RelativePositionEmbedding(
          hidden_size=self.params['hidden_size']
      )
    self.encoder_stack = encoder_stack.EncoderStack(params)
    self.fc1 = tf.keras.layers.Dense(
        units=params['vocab_size'],
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
    )
    if self.params['ploidy'] == 2:
      self.fc2 = tf.keras.layers.Dense(
          units=params['vocab_size'],
          activation=None,
          use_bias=True,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
      )
    self.softmax = tf.keras.layers.Softmax()

  def get_config(self) -> Dict[str, Any]:
    return {
        'params': self.params,
    }

  def call(
      self,
      inputs: tf.Tensor,
      training: bool = True,
      mask: Union[Optional[tf.Tensor], Any] = None,
  ) -> tf.Tensor:
    """Runs a forward pass of the model.

    Args:
      inputs: tensor of shape (batch_size, hidden_size, input_length
        num_channels).
      training: boolean, whether in training mode or not.
      mask: A tensor representing the mask.

    Returns:
      Output from softmax layer, which is a distribution over the vocabulary at
      each position in the sequence.
    """
    with tf.name_scope('Transformer'):
      intermediate_outputs_dict = self.get_intermediate_outputs(
          inputs, training=training
      )
      logits = intermediate_outputs_dict['logits']
      preds = self.softmax(logits)

      if self.params['ploidy'] == 2:
        logits2 = intermediate_outputs_dict['logits2']
        preds2 = self.softmax(logits2)
        # Stack, e.g. if both are (32, 100, 5), they stack to (32, 2, 100, 5).
        stacked = tf.stack([preds, preds2], axis=1)
        return stacked
      else:
        return preds

  def get_intermediate_outputs(
      self, inputs: tf.Tensor, training: bool
  ) -> Dict[str, tf.Tensor]:
    """Get intermediate outputs of the model.

    Args:
      inputs: tensor of shape (batch_size, hidden_size, input_length
        num_channels).
      training: boolean, whether in training mode or not.

    Returns:
      Dictionary with the following (key:value) pairs:
        "self_attention_layer_{n}": Attention layer output for every layer in
        the encoder stack with shape [batch_size, input_length, hidden_size].
        "attention_scores_{n}" : Attention map for every layer in the
        encoder stack with shape [batch_size, num_heads, input_length,
        input_length].
        "ffn_layer_{n}": Feedforward network output for every layer in the
        encoder stack with shape [batch_size, input_length, hidden_size].
        "final_output": Final output of the entire encoder stack after
        normalization with shape [batch_size, input_length, hidden_size]. Used
        as input to the fully-connected layer which outputs logits.
        "logits": Logits over the vocabulary at each position in the sequence
        with shape [batch_size, input_length, vocab_size].
    """

    # Get rid of the channel dimension as we only have one channel.

    # logging.info('DEBUG MODEL: INPUT SIZE IS: ' + str(inputs.shape))
    # logging.info('DEBUG MODEL: INPUT TYPE: ' + str(type(inputs)))
    inputs = tf.squeeze(inputs, -1)
    # `inputs` is of shape (batch_size, hidden_size, input_length). For the
    # Transformer, we need to change the format to be the following:
    # (batch_size, input_length, hidden_size).
    inputs = tf.transpose(inputs, [0, 2, 1])

    # Attention_bias for our model should be all 0s with shape
    # (batch_size, 1, 1, input_length). See model_utils.get_padding_bias
    # to see how this is calculated in the base model.
    all_zeros = tf.reduce_sum(tf.zeros_like(inputs), -1)
    attention_bias = tf.expand_dims(tf.expand_dims(all_zeros, 1), 1)

    # Run inputs through the encoder. Encoder returns a dictionary of
    # logits from dense layer as well as other intermediate model outputs.
    intermediate_outputs_dict = self.encode(inputs, attention_bias, training)
    return intermediate_outputs_dict

  def encode(
      self, inputs: tf.Tensor, attention_bias: tf.Tensor, training: bool
  ) -> Dict[str, tf.Tensor]:
    """Runs the input through Encoder stack and problem-specific layers."""

    with tf.name_scope('encode'):
      # The input for each position is already a vector, so we do not use
      # embeddings here, unlike the base model. Base model input is a token at
      # each position, which must first be embedded as a vector. In the future,
      # we may want to use embeddings for part of the input, such as the bases,
      # so that we can learn the scale of values.
      encoder_inputs = inputs

      # Positional embedding only works when we have an even value for the
      # hidden_size. If hidden_size is odd, add an empty row to make it even.
      if self.params.add_pos_encoding and encoder_inputs.shape[2] % 2 != 0:
        empty_row = tf.zeros(
            shape=(encoder_inputs.shape[0], encoder_inputs.shape[1], 1)
        )
        encoder_inputs = tf.concat([encoder_inputs, empty_row], axis=-1)
        assert self.params.hidden_size == encoder_inputs.shape[2]

      # All values in `input_padding` should be 0 and shape should be
      # (batch_size, input_length). See model_utils.get_padding to see how this
      # is computed for the base model.
      inputs_padding = tf.reduce_sum(tf.zeros_like(encoder_inputs), -1)

      # Cast input `attention_bias` to correct type, as done in the base model.
      attention_bias = tf.cast(
          attention_bias, data_providers.get_tf_data_type()
      )

      # Add positional encoding to the input. The scale of the positional
      # encoding relative to the input values will matter since we are not
      # learning the input embedding.
      if self.params['add_pos_encoding']:
        with tf.name_scope('add_pos_encoding'):
          pos_encoding = self.position_embedding(inputs=encoder_inputs)
          pos_encoding = tf.cast(
              pos_encoding, data_providers.get_tf_data_type()
          )
          encoder_inputs += pos_encoding

      # Add dropout when training.
      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params['layer_postprocess_dropout']
        )

      # Pass inputs through the encoder. As mentioned above, `inputs_padding` is
      # not actually used by EncoderStack.call. Encoder stack output is a
      # dictionary containing final output of the encoder stack with shape
      # (batch_size, input_length, hidden_size) as well as intermediate outputs
      # of each of the attention and feed forward network layers in the stack.
      encoder_outputs_dict = self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training
      )

      # Pass the final output of the encoder stack through dense layer and
      # output logits over vocab for each position.
      encoder_outputs = self.fc1(encoder_outputs_dict['final_output'])
      # Add logits to the outputs dictionary.
      encoder_outputs_dict['logits'] = encoder_outputs
      if self.params['ploidy'] == 2:
        second_output = self.fc2(encoder_outputs_dict['final_output'])
        encoder_outputs_dict['logits2'] = second_output
      return encoder_outputs_dict

  def decode(
      self,
      encoder_outputs: tf.Tensor,
      attention_bias: tf.Tensor,
      training: bool,
  ) -> tf.Tensor:
    """Returns the outputs from the encoder."""

    raise NotImplementedError

  def predict(self, encoder_inputs: tf.Tensor) -> tf.Tensor:
    """Returns the argmax of the decoder output, which comes from a softmax."""

    # The base model also has a predict method that behaves differently. This
    # predict function is consistent with how predict behaves for other
    # DeepConsensus models (conv, FC), but we may want to change this in the
    # future to match the transformer base class. For more details, see:
    # https://github.com/tensorflow/models/blob/bc71d8e9e155d34a38af8489ad4cbb2fde6fa152/official/nlp/transformer/transformer.py#L279
    return self.call(encoder_inputs, training=False)


class EncoderOnlyLearnedValuesTransformer(EncoderOnlyTransformer):
  """Modified transformer that learns embeddings for the bases."""

  def __init__(
      self, params: ml_collections.ConfigDict, name: Optional[str] = None
  ):
    super().__init__(params, name=name)
    vocab_size_by_feature_name = encoding.get_max_encoding_value_by_feature()
    hidden_size_by_feature_name = model_configs.get_feature_hidden_size_map()
    self.embedding_layers = {}
    for feature_name, vocab_size in vocab_size_by_feature_name.items():
      self.embedding_layers[feature_name] = ModifiedOnDeviceEmbedding(
          vocab_size=vocab_size + 1,
          embedding_width=hidden_size_by_feature_name[feature_name],
          name=f'{feature_name}_embedding',
      )

    # Define a dense layer to linearly map the concatenated embeddings of
    # all subreads at a given position to a smaller dimension
    # (transformer_input_size) in order to keep the transformer layers small.
    if self.params.condense_transformer_input:
      logging.info('Condensing input.')
      self.transformer_input_condenser = tf.keras.layers.Dense(
          units=(params.transformer_input_size),
          activation=None,
          use_bias=False,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
      )

  def encode(
      self, inputs: tf.Tensor, attention_bias: tf.Tensor, training: bool
  ) -> Dict[str, tf.Tensor]:
    """Runs the input through Encoder stack and problem-specific layers."""
    # Input to embedding layer is [batch_size, length] and output will be
    # [batch_size, length, embedding_size]. Embed each row of the input
    # separately and then concatenate.
    embedded_inputs = []
    indices_by_feature_name = data_providers.get_feature_indices()
    for feature_name, indices in indices_by_feature_name.items():
      if feature_name not in self.embedding_layers:
        raise ValueError(f'{feature_name} not found in embedding layers')
      for i in range(*indices):
        embedded = self.embedding_layers[feature_name](
            tf.cast(inputs[:, :, i], tf.int32)
        )
        embedded_inputs.append(embedded)

    embedded_inputs = tf.concat(embedded_inputs, axis=-1)
    embedded_inputs = tf.cast(
        embedded_inputs, data_providers.get_tf_data_type()
    )

    if self.params.condense_transformer_input:
      # Condense the transformer input at each position to a smaller vector to
      # reduce the transformer hidden size, since the transformer model size is
      # quadratic in its hidden size.
      # Shape: [batch_size, length, transformer_input_size]
      transformer_input = self.transformer_input_condenser(embedded_inputs)
    else:
      transformer_input = embedded_inputs

    return super().encode(transformer_input, attention_bias, training)
