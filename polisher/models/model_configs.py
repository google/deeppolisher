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
"""Architecture and training hyperparameters for networks."""
# pylint: disable=line-too-long
import os

from typing import Optional, Dict
import ml_collections
from polisher.make_images import encoding

# Do not add any additional imports to the config.
# It can lead to circular dependencies easily and should not be necessary
# for setting parameters.


def testdata_directory() -> str:
  curr_dir = os.path.dirname(__file__)
  testdata_path = os.path.join(curr_dir, '../testdata/tf_examples')
  return testdata_path


def get_feature_hidden_size_map() -> Dict[str, int]:
  """Get hidden size defined for each feature."""
  feature_depths = encoding.get_feature_depths()
  feature_hidden_sizes = {
      'reference': 2,
      'encoded_bases': 8,
      'encoded_match_mismatch': 8,
      'encoded_base_qualities': 8,
      'encoded_mapping_quality': 8,
      'encoded_hp_tag': 2,
  }
  features_with_hidden_size = feature_hidden_sizes.keys()
  features_in_example = feature_depths.keys()
  for feature in features_in_example:
    if feature not in features_with_hidden_size:
      raise ValueError(f'Hidden size for {feature!r} is not defined.')

  return feature_hidden_sizes


def get_total_hidden_size() -> int:
  """Get total hidden size."""
  feature_depths = encoding.get_feature_depths()
  feature_hidden_sizes = get_feature_hidden_size_map()

  total_hidden_size = 0
  for feature, depth in feature_depths.items():
    total_hidden_size += feature_hidden_sizes[feature] * depth

  return total_hidden_size


############### Base params for different model architectures ###############
def _set_transformer_hparams(params):
  """Updates given config with values for the Transformer model."""
  # Architecture
  params.model_name = 'transformer_learn_values'
  params.add_pos_encoding = True
  # Num heads should be divisible by hidden size. This value should be tuned for
  # the production setting.
  params.num_heads = 2
  params.layer_norm = False
  params.condense_transformer_input = True
  params.transformer_model_size = 'base'

  params.num_channels = 1
  params.hidden_size = get_total_hidden_size()
  params.transformer_input_size = 280

  # Dropout values (only used when training).
  params.layer_postprocess_dropout = 0.1
  params.attention_dropout = 0.1
  params.relu_dropout = 0.1

  # Training in Xmanager setup
  params.batch_size = 32
  params.num_epochs = 20000
  params.num_epochs_for_decay = 20000
  params.buffer_size = 10000

  params.initial_learning_rate = 3.6246e-3
  params.end_learning_rate = 2.86594e-5
  params.warmup_steps = 35536
  params.weight_decay_rate = 6.9868e-3
  params.beta_1 = 0.9
  params.beta_2 = 0.999
  params.epsilon = 1e-6
  params.ploidy = 1


def _set_transformer_2outputs_hparams(params):
  # Start from regular params
  _set_transformer_hparams(params)
  params.loss_function = 'diploid_alignment_loss'
  params.ploidy = 2


############### Base params for different datasets ###############


############### Core function for setting all config values ###############


def get_config(config_name: Optional[str] = None) -> ml_collections.ConfigDict:
  """Returns the default configuration as instance of ConfigDict.

  Valid config names must consist of two parts: {model_name}+{dataset_name}. The
  "+" must be present as a separator between the two parts. For example,
  transformer_learn_values+HG002 is a valid name.

  Valid model names include:
    * transformer_learn_values
    * transformer_2output

  Valid dataset names include:
    * HG002
    * diploid_HG002
    * test

  Args:
    config_name: String consisting of two parts, model name and dataset name,
      separated by a "+".

  Returns:
    A config dictionary containing the valid configs for the model and dataset
    specified.
  """
  params = ml_collections.ConfigDict()
  # Used for generating replicates.
  params.trial = 1

  # Specify common configs here.
  params.vocab_size = encoding.get_vocab_size()
  params.tensorboard_update_freq = 'batch'
  params.model_checkpoint_freq = 'epoch'
  params.seed = 1
  params.remove_label_gaps = False
  params.loss_function = 'alignment_loss'

  # AlignmentLoss parameters
  params.del_cost = 10.0
  params.loss_reg = 0.1
  params.band_width = None

  # Default model and dataset
  params.model_config_name = 'transformer_learn_values'
  params.dataset_config_name = 'HG002'

  # Scaling factor to multiply the batch_size when using TPUs since they have
  # more memory than GPUs.
  params.tpu_scale_factor = 1

  # Allow for a base config to be loaded when no config_name is passed.
  if config_name is None:
    return params

  model_config_name, dataset_config_name = config_name.split('+')
  params.model_config_name = model_config_name
  params.dataset_config_name = dataset_config_name
  params.tf_dataset = None
  params.limit = -1

  if model_config_name == 'transformer_learn_values':
    _set_transformer_hparams(params)
  elif model_config_name == 'transformer_2output':
    _set_transformer_2outputs_hparams(params)
  else:
    raise ValueError('Unknown model_config_name: %s' % model_config_name)

  if dataset_config_name in ['HG002', 'haploid']:
    # Keeping "HG002" here for backwards compatibility.
    _set_data_hparams_for_haploid(params)
  elif dataset_config_name == 'haploid_mini':
    _set_data_hparams_for_haploid_mini(params)
  elif dataset_config_name == 'diploid':
    _set_data_hparams_for_diploid(params)
  elif dataset_config_name == 'diploid_mini':
    _set_data_hparams_for_diploid_mini(params)
  else:
    raise ValueError(
        f'dataset_config_name is {dataset_config_name}. '
        'Must be one of the above. See get_config() for options.'
    )
  return params
