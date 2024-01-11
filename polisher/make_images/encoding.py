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
"""Encoding methods for make examples."""
from typing import List, Dict

# ENCODINGS
# Base encoding of gap ('*') and pad (' ') is considered the same
# because of their use in lossess_and_metrics.py method.
_BASE_ENCODINGS = {'*': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
_VALID_BASES = ['A', 'C', 'G', 'T']
_GAP_TOKEN = '*'
_MAPPING_QUALITY_CAP = 100
_BASE_QUALITY_CAP = 100
_MATCH_MISMATCH_ENCODINGS = {'M': 1, 'X': 2}
_HAPLOTYPE_TAG_ENCODING = {0: 1, 1: 2, 2: 3}


_MIN_MAPPING_QUALITY = 1
_TRUTH_MIN_MAPPING_QUALITY = 30
_WINDOW_LENGTH = 100
_OVERLAP_LENGTH = 20
_MAX_COVERAGE_PER_HAPLOTYPE = 30
_PADDING_FOR_REFERENCE_SEQUENCE = 10
_ACTIVE_POSITION_DISTANCE_THRESHOLD = 50
_ACTIVE_POSITION_FREQ_THRESHOLD = 0.05
_ACTIVE_POSITION_MIN_SUPPORT = 2
_MAX_ALLOWED_INDEL_LENGTH = 50
_MAX_ALLOWED_READS_IN_REGION = 1000


def get_min_mapping_quality() -> int:
  """Returns the min mapping quality value."""
  return _MIN_MAPPING_QUALITY


def get_truth_min_mapping_quality() -> int:
  """Returns the min mapping quality value."""
  return _TRUTH_MIN_MAPPING_QUALITY


def get_window_length() -> int:
  """Returns the window length value."""
  return _WINDOW_LENGTH


def get_overlap_length() -> int:
  """Returns the overlap length value."""
  return _OVERLAP_LENGTH


def get_max_coverage_per_haplotype() -> int:
  """Returns the max coverage per haplotype value."""
  return _MAX_COVERAGE_PER_HAPLOTYPE


def get_padding_for_reference_sequence() -> int:
  """Returns the padding for reference sequence value."""
  return _PADDING_FOR_REFERENCE_SEQUENCE


def get_active_position_distance_threshold() -> int:
  """Returns the active position distance threshold value."""
  return _ACTIVE_POSITION_DISTANCE_THRESHOLD


def get_active_position_freq_threshold() -> float:
  """Returns the active position freq threshold value."""
  return _ACTIVE_POSITION_FREQ_THRESHOLD


def get_active_position_min_support() -> int:
  """Returns the active position min support value."""
  return _ACTIVE_POSITION_MIN_SUPPORT


def get_max_allowed_indel_length() -> int:
  """Returns the max allowed indel length value."""
  return _MAX_ALLOWED_INDEL_LENGTH


def get_max_allowed_reads_in_region() -> int:
  """Returns the max allowed reads in region value."""
  return _MAX_ALLOWED_READS_IN_REGION


def get_valid_bases() -> List[str]:
  """Returns the valid bases."""
  return _VALID_BASES


def get_feature_depths() -> Dict[str, int]:
  """Returns the set of features used and the coverage per feature."""
  feature_rows = {
      'reference': 1,
      'encoded_bases': _MAX_COVERAGE_PER_HAPLOTYPE,
      'encoded_match_mismatch': _MAX_COVERAGE_PER_HAPLOTYPE,
      'encoded_base_qualities': _MAX_COVERAGE_PER_HAPLOTYPE,
      'encoded_mapping_quality': _MAX_COVERAGE_PER_HAPLOTYPE,
  }
  return feature_rows


def get_max_length() -> int:
  """Return maximum length of an example."""
  return _WINDOW_LENGTH


def get_max_coverage() -> int:
  """Return maximum coverage of an example."""
  return _MAX_COVERAGE_PER_HAPLOTYPE


def get_max_encoding_value_by_feature() -> Dict[str, int]:
  """Return maximum value used to represent each feature."""
  max_encoding_value = {
      'reference': max(_BASE_ENCODINGS.values()),
      'encoded_bases': max(_BASE_ENCODINGS.values()),
      'encoded_match_mismatch': max(_MATCH_MISMATCH_ENCODINGS.values()),
      'encoded_base_qualities': _BASE_QUALITY_CAP,
      'encoded_mapping_quality': _MAPPING_QUALITY_CAP,
      'encoded_hp_tag': max(_HAPLOTYPE_TAG_ENCODING.values()),
  }
  return max_encoding_value


def get_vocab_size() -> int:
  """Get vocab size.

  Vocab size is the bases we encode excluding the gap token.

  Returns:
  Length of the _BASE_ENCODINGS minus the gap token.
  """
  return len(_BASE_ENCODINGS.items())


def get_vocab() -> List[str]:
  """Return the prediction vocabulary used."""
  return list(_BASE_ENCODINGS.keys())


def get_gap_token() -> str:
  """Return the prediction vocabulary used."""
  return _GAP_TOKEN


def get_gap_or_pad_encoding() -> int:
  """Get gap or pad encoding value.

  Returns:
  Encoding value of gap (*) bases.
  """
  return _BASE_ENCODINGS['*']


def get_nucleotide_encoding(nucleotide_base: str) -> int:
  """Get base encoding of a given nucleotide.

  Args:
    nucleotide_base: A base observed in a read.

  Returns:
    An encoding value that represents the base.
  """
  if nucleotide_base.upper() not in _BASE_ENCODINGS:
    return 0
  return _BASE_ENCODINGS[nucleotide_base.upper()]


def get_base_quality_encoding(base_quality: int) -> int:
  """Get base quality encoding.

  We do not scale base quality, rather use the original value. We cap it to
  ensure that the value doesn't overflow. This capping is at Q100 which is
  very large in log space.

  Args:
    base_quality: A base quality value.

  Returns:
    A capped base_quality value.
  """
  return int(min(base_quality, _BASE_QUALITY_CAP))


def get_mapping_quality_encoding(mapping_quality: int) -> int:
  """Get mapping quality encoding.

  We do not scale mapping quality, rather use the original value. We cap it to
  ensure that the value doesn't overflow. This capping is at Q100 which is
  very large in log space.

  Args:
    mapping_quality: A base quality value.

  Returns:
    A capped mapping_quality value.
  """
  return int(min(mapping_quality, _MAPPING_QUALITY_CAP))


def get_hp_encoding(haplotype_tag: int) -> int:
  """Get haplotype tag encoding.

  Args:
    haplotype_tag: A haplotype tag.

  Returns:
    An encoding of the haplotype tag.
  """
  return _HAPLOTYPE_TAG_ENCODING[haplotype_tag]


def get_match_mismatch_encoding(match: bool) -> int:
  """Get match/mismatchmat encoding.

  Args:
    match: A boolean value. True if it's a match.

  Returns:
    An encoding of the match/mismatch.
  """
  if match:
    return _MATCH_MISMATCH_ENCODINGS['M']
  return _MATCH_MISMATCH_ENCODINGS['X']
