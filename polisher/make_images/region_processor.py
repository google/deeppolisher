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
"""Classes to process region for make examples."""

import dataclasses
import logging
import os
from typing import Any, Optional, Sequence

import intervaltree
import numpy as np
import pysam
import tensorflow as tf

from polisher.make_images import encoding
from polisher.make_images import utils_make_images
from polisher.make_images.utils_make_images import RegionRecord


def get_feature_depths() -> dict[str, int]:
  """Returns the set of features used and the coverage per feature."""
  feature_rows = {
      'reference': 1,
      'encoded_bases': encoding.get_max_coverage_per_haplotype(),
      'encoded_match_mismatch': encoding.get_max_coverage_per_haplotype(),
      'encoded_base_qualities': encoding.get_max_coverage_per_haplotype(),
      'encoded_mapping_quality': encoding.get_max_coverage_per_haplotype(),
  }
  return feature_rows


@dataclasses.dataclass
class PositionalSpacingRecord:
  """Represents positional spacing information due to insertion of a region.

  We use this class to aggregate positional information before we perform
  any spacing. This takes in reads of a region and calculates maximum length of
  insert observed at a position to calculate how much spacing is required.

  We also record the read span in an interval tree so we can query it to find
  any read that spans an active position.

  Attributes:
    max_observed_insert: Max insert length observed at each position.
    region_length_with_spacing: Total length of region after spacing.
    read_span_intervaltree: An interval tree that contains the ranges of the
      reads.
  """
  max_observed_insert: np.ndarray
  region_length_with_spacing: int
  read_span_intervaltree: intervaltree.IntervalTree

  def __init__(self, interval: RegionRecord, reads: list[pysam.AlignedSegment]):
    """Initializes for a region with reads aligned to the region.

    Arguments:
      interval: RegionRecord representing the interval.
      reads: Aligned reads within the interval.
    """
    region_length = get_region_length(interval)
    self.max_observed_insert = np.zeros(region_length, dtype=np.int32)
    self.region_length_with_spacing = 0
    self.read_span_intervaltree = intervaltree.IntervalTree()
    for read in reads:
      current_ref_pos = read.reference_start
      current_read_index = 0
      match_observed = False
      read_end = read.reference_start

      for cigar_op, cigar_len in read.cigartuples:
        # we have skipped past the window, no need to process this read anymore
        if current_ref_pos > interval.stop:
          break
        # The match/diff cases don't require any spacing, so we only update
        # the non-match statistics here.
        if cigar_op in [pysam.CMATCH, pysam.CDIFF, pysam.CEQUAL]:
          # Matches/Mismatches do not require spacing.
          match_observed = True
          current_read_index += cigar_len
          current_ref_pos += cigar_len
        elif cigar_op in [pysam.CSOFT_CLIP, pysam.CINS]:
          # Inserts require spacing so we update the max observed insert
          # record against the obsered insert of length cigar_len
          if (
              interval.start <= current_ref_pos <= interval.stop
              and match_observed
              and cigar_len <= encoding.get_max_allowed_indel_length()
          ):
            region_index = current_ref_pos - interval.start - 1
            self.max_observed_insert[region_index] = max(
                self.max_observed_insert[region_index], cigar_len
            )
          current_read_index += cigar_len
        elif cigar_op in [pysam.CREF_SKIP, pysam.CDEL]:
          # Deletes do not require spacing so we simply move
          current_ref_pos += cigar_len

        # update read end position
        read_end = max(read_end, current_ref_pos)
      # Don't add truth_read to interval_tree as it will end up in the
      # example pileup which is undesirable.
      if read.query_name != 'truth_read':
        read_span = intervaltree.Interval(read.reference_start, read_end,
                                          read.query_name)
        self.read_span_intervaltree.add(read_span)
    total_insert_spacings = np.sum(self.max_observed_insert)
    self.region_length_with_spacing = region_length + total_insert_spacings


@dataclasses.dataclass
class PositionalNonmatchRecord:
  """Represents positional nonmatch information of a region.

  This class is used to aggregate positional non-match information.
  This takes in reads of a region and calculates the depth and
  non-match (SNP+INDEL) rate to find active positions that we can use to create
  windows.

  Attributes:
    observed_depth: Number of total reads observed at each position.
    observed_nonmatches: Number of SNP/INDEL found at each position.
    observed_nonmatch_frequencies: Non-match freq. at each position of the
      region defined as (non_matches/depth).
    active_positions: Genomics position where mismatch frequency is above a
      defined threshold.
  """
  observed_depth: np.ndarray
  observed_nonmatches: np.ndarray
  observed_nonmatch_frequencies: np.ndarray
  active_positions: list[int]

  def __init__(
      self,
      interval: RegionRecord,
      reads: list[pysam.AlignedSegment],
      reference_sequence: str,
  ):
    """Initializes class for a region of length region_length.

    Arguments:
      interval: RegionRecord representing the interval.
      reads: Aligned reads within the interval.
      reference_sequence: Reference sequence of the region.
    """
    region_length = get_region_length(interval)
    self.observed_depth = np.zeros(region_length, dtype=np.int32)
    self.observed_nonmatches = np.zeros(region_length, dtype=np.int32)
    self.observed_nonmatch_frequencies = np.zeros(
        region_length, dtype=np.float32)
    self.active_positions = []

    for read in reads:
      current_ref_pos = read.reference_start
      current_read_index = 0
      match_observed = False

      for cigar_op, cigar_len in read.cigartuples:
        if current_ref_pos > interval.stop:
          break
        # The match/diff cases don't require any spacing, so we only update
        # the non-match statistics here.
        if cigar_op in [pysam.CMATCH, pysam.CDIFF, pysam.CEQUAL]:
          for _ in range(0, cigar_len):
            # Mismatches can occur within these CIGAR operations, we have to
            # unroll the entire cigar and check each base if they are match
            # to the reference.
            if interval.start <= current_ref_pos <= interval.stop:
              region_index = current_ref_pos - interval.start
              read_base = read.query_sequence[current_read_index]
              ref_base = reference_sequence[region_index]
              # if the read and reference are difference bases then
              # we have observed a nonmatch.
              if read_base != ref_base:
                self.observed_nonmatches[region_index] += 1
              # update the depth record each time.
              self.observed_depth[region_index] += 1
              match_observed = True
            current_read_index += 1
            current_ref_pos += 1
        elif cigar_op in [pysam.CSOFT_CLIP, pysam.CINS]:
          # Inserts are always non-matches so update the nonmatch count.
          if (
              interval.start <= current_ref_pos <= interval.stop
              and match_observed
              and cigar_len <= encoding.get_max_allowed_indel_length()
          ):
            region_index = current_ref_pos - interval.start - 1
            self.observed_nonmatches[region_index] += 1
          current_read_index += cigar_len

        elif cigar_op in [pysam.CREF_SKIP, pysam.CDEL]:
          # Deletes are always non-matches so update the count.
          # the region_index for deletes are current_ref_pos - 1 for the anchor.
          # however, observed depth for deletes we iterate over
          # each position and update the depth we observed
          for _ in range(0, cigar_len):
            region_index = current_ref_pos - interval.start
            if (interval.start <= current_ref_pos <= interval.stop and
                match_observed):
              if cigar_len <= encoding.get_max_allowed_indel_length():
                self.observed_nonmatches[region_index] += 1
              # We should increase depth even if the del length is large
              # but it will not counted as a mismatch.
              self.observed_depth[region_index] += 1
            current_ref_pos += 1

    # Once all the nonmatches are counted, we can count the frequency and
    # active positions
    self.active_positions = []
    for i in range(0, region_length):
      self.observed_nonmatch_frequencies[i] = self.observed_nonmatches[i] / max(
          1.0, self.observed_depth[i]
      )
      if (
          self.observed_nonmatch_frequencies[i]
          >= encoding.get_active_position_freq_threshold()
          and self.observed_nonmatches[i]
          >= encoding.get_active_position_min_support()
      ):
        self.active_positions.append(i + interval.start)


@dataclasses.dataclass
class SpacedReferenceRecord:
  """Represents spaced reference sequence and positional information.

  After spacing we want to keep track of the genomic coordinates and where
  the spacing is happening. We use two values to do that, one is the reference
  position and other one is the reference index. The position represents the
  genomic position or anchor position in case of an insert. The indices keep
  track of the index value of the inserts. For example:

  Before spacing:
  Genomic position: 12345
  Genomic sequence: ACGTC

  After spacing it becomes:
  Genomic position: 1234445
  Genomic sequence: ACGT**C
  Genomic index:    0000120

  So we use position,index pair to keep track of genomic locations.

  Finally, we use the reference_position_to_index dictionary to look-up index
  for each active position for which we generate an example.

  Attributes:
    reference_positions: Genomic position of reference coordinates after
      spacing.
    reference_indices: Indices indicating the spacing extension of genomic
      positions.
    ref_position_to_index: A map between reference position and region index.
    spaced_reference_sequence: Reference sequence with spacing.
  """
  reference_positions: np.ndarray
  reference_indices: np.ndarray
  ref_position_to_index: dict[int, int]
  spaced_reference_sequence: str

  def __init__(self, interval: RegionRecord,
               spacing_info: PositionalSpacingRecord, reference_sequence: str):
    """Initializes based on observed spacing due to insertions.

    Arguments:
      interval: A genomic interval.
      spacing_info: Record with spacing information.
      reference_sequence: Reference sequence of the region.
    """
    region_length = get_region_length(interval)
    self.reference_positions = np.zeros(
        spacing_info.region_length_with_spacing, dtype=np.int32)
    self.reference_indices = np.zeros(
        spacing_info.region_length_with_spacing, dtype=np.int32)
    self.ref_position_to_index = {}
    self.spaced_reference_sequence = ''

    spaced_reference_bases = list()
    spaced_index = 0
    for pos_index in range(0, region_length):
      # update the main positions/reference positions
      self.reference_positions[spaced_index] = interval.start + pos_index
      self.reference_indices[spaced_index] = 0
      spaced_reference_bases.append(reference_sequence[pos_index])
      spaced_index += 1

      # now see if there's an insert and update the positions accordingly
      for ins_index in range(0, spacing_info.max_observed_insert[pos_index]):
        spaced_reference_bases.append('*')
        self.reference_positions[spaced_index] = interval.start + pos_index
        self.reference_indices[spaced_index] = ins_index + 1
        spaced_index += 1

    # create a dictionary to go back and forth between reference position and
    # regional indices
    for index, ref_pos in enumerate(self.reference_positions):
      if ref_pos not in self.ref_position_to_index:
        self.ref_position_to_index[ref_pos] = index
    self.spaced_reference_sequence = ''.join(spaced_reference_bases)


class EncodedRead:
  """Represents an encoded read.

  This class is used to encode a read after we have spaced out a region.

  Attributes:
    encoded_bases: Base encoding of the read sequence.
    encoded_base_qualities: Base quality value encoding of read sequence.
    encoded_mapping_quality: Mapping quality value encoding.
    encoded_match_mismatch: Encoding if the observed base is a match/mismatch.
  """
  encoded_bases: np.ndarray
  encoded_base_qualities: np.ndarray
  encoded_mapping_quality: np.ndarray
  encoded_match_mismatch: np.ndarray

  def __init__(self, interval: RegionRecord, read: pysam.AlignedSegment,
               spaced_reference_record: SpacedReferenceRecord,
               spaced_length: int):
    """Initialize by encoding a read.

    Args:
      interval: Region interval for the read.
      read: The read that will be encoded.
      spaced_reference_record: Reference with spacing information.
      spaced_length: Length of region after spacing.
    """
    self.encoded_bases = np.zeros(spaced_length, dtype=np.uint8)
    self.encoded_base_qualities = np.zeros(spaced_length, dtype=np.uint8)
    self.encoded_mapping_quality = np.zeros(spaced_length, dtype=np.uint8)
    self.encoded_match_mismatch = np.zeros(spaced_length, dtype=np.uint8)

    current_ref_pos = read.reference_start
    current_read_index = 0
    # We use match_observed to keep track of when we first see a match
    # so we can enable insert encoding. We don't encode inserts without any
    # anchors so this anchors that the anchor base is present for an insertion.
    match_observed = False

    # iterate over the read to find the maximum insert size
    # we observe at any position within the window.
    for cigar_op, cigar_len in read.cigartuples:
      # we have skipped past the window, no need to process this read anymore
      if current_ref_pos > interval.stop:
        break
      # if it's a match then only move forward, matches don't need spacing
      if cigar_op in [pysam.CMATCH, pysam.CDIFF, pysam.CEQUAL]:
        for i in range(0, cigar_len):
          # the base is within the window
          if interval.start <= current_ref_pos <= interval.stop:
            region_index = spaced_reference_record.ref_position_to_index[
                current_ref_pos]
            self.encode_read_base_attributes(spaced_reference_record,
                                             region_index, current_read_index,
                                             read)
            match_observed = True
          current_read_index += 1
          current_ref_pos += 1
      # if it's an insert then we record it
      elif cigar_op in [pysam.CSOFT_CLIP, pysam.CINS]:
        # Make sure the insert is within the window and make sure
        # the insert/soft-clip is anchored with a match. We don't allow
        # for left spacing at this moment.
        if interval.start <= current_ref_pos <= interval.stop:
          for i in range(0, cigar_len):
            # the base is within the window
            if (
                cigar_op == pysam.CINS
                and match_observed
                and cigar_len <= encoding.get_max_allowed_indel_length()
            ):
              region_index = spaced_reference_record.ref_position_to_index[
                  current_ref_pos - 1] + i + 1
              self.encode_read_base_attributes(spaced_reference_record,
                                               region_index, current_read_index,
                                               read)
            current_read_index += 1
        else:
          # roll the read index when the position is outside the interval range
          current_read_index += cigar_len
      elif cigar_op in [pysam.CREF_SKIP, pysam.CDEL]:
        for i in range(0, cigar_len):
          # the base is within the window
          if (interval.start <= current_ref_pos <= interval.stop and
              match_observed):
            region_index = spaced_reference_record.ref_position_to_index[
                current_ref_pos]
            self.encode_read_base_attributes(
                spaced_reference_record,
                region_index,
                current_read_index,
                read,
                is_del=True)
          current_ref_pos += 1

  def encode_read_base_attributes(
      self,
      spaced_reference_record: SpacedReferenceRecord,
      region_index: int,
      read_index: int,
      read: pysam.AlignedSegment,
      is_del: bool = False):
    """Encode a base of the read to encoded attribute.

    Args:
      spaced_reference_record: A record with reference spacing information.
      region_index: Region index for this base.
      read_index: Read index for this base.
      read: Read to which the read_base belongs to.
      is_del: True if the base we are encoding is a deletion.
    """
    # fetch the read attributes
    if not is_del:
      read_base = read.query_sequence[read_index]
    else:
      read_base = '*'

    if read.query_qualities is not None and not is_del:
      read_base_quality = read.query_qualities[read_index]
    else:
      read_base_quality = 0
    ref_base = spaced_reference_record.spaced_reference_sequence[region_index]

    # encode the values we fetched above and set them to corresponding values.
    self.encoded_bases[region_index] = encoding.get_nucleotide_encoding(
        read_base)
    self.encoded_base_qualities[
        region_index] = encoding.get_base_quality_encoding(read_base_quality)
    self.encoded_mapping_quality[
        region_index] = encoding.get_mapping_quality_encoding(
            read.mapping_quality)

    self.encoded_match_mismatch[
        region_index] = encoding.get_match_mismatch_encoding(
            match=(read_base == ref_base))


class EncodedReference:
  """Represents an encoded reference sequence."""
  encoded_reference: np.ndarray

  def __init__(self, spacing_record: PositionalSpacingRecord,
               spaced_reference_record: SpacedReferenceRecord,
               spaced_length: int):
    ref_bases = list(spaced_reference_record.spaced_reference_sequence)
    # the encoded reference length needs to be at least the size of the window
    self.encoded_reference = np.zeros(spaced_length, dtype=np.uint8)

    for idx, base in enumerate(ref_bases):
      if idx >= spacing_record.region_length_with_spacing:
        break
      self.encoded_reference[idx] = encoding.get_nucleotide_encoding(base)


class Example:
  """Example of a region generated using encoded reads and reference.

  Attributes:
    max_coverage: Maximum coverage allowed for the example.
    example_width: Width of the example (in this case region).
    reads: A list of encoded reads.
    read_features: A list of features to use to encode examples.
    feature_rows: A dictionary with how many rows each feature would consume.
    feature_indices: Row indices for each feature set.
    dims: Dimenstion of the example.
    example: An example representing stacked encoded features.
  """

  read_features = [
      'encoded_bases', 'encoded_base_qualities', 'encoded_mapping_quality',
      'encoded_match_mismatch'
  ]

  def __init__(self, max_coverage: int, example_width: int,
               reads: list[EncodedRead], encoded_ref: EncodedReference):
    self.max_coverage = max_coverage
    self.example_width = example_width
    self.reads = reads
    self.feature_rows = {
        'reference': 1,
        'encoded_bases': max_coverage,
        'encoded_match_mismatch': max_coverage,
        'encoded_base_qualities': max_coverage,
        'encoded_mapping_quality': max_coverage,
    }
    # Sets slices indicating rows for each feature type.
    self.feature_indices = dict()
    row_index = 0
    for feature_name, feature_depth in self.feature_rows.items():
      self.feature_indices[feature_name] = slice(
          row_index, row_index + self.feature_rows[feature_name])
      setattr(self, feature_name, row_index)
      row_index += feature_depth

    # generate the example
    self.dims = (self.tensor_height, example_width)
    self.example = np.zeros(shape=self.dims, dtype=np.uint8)

    # We encode the top row as the reference.
    self.example[
        self.feature_indices['reference']] = encoded_ref.encoded_reference

    # Then we loop over each feature and stack them to create the example.
    for feature in self.read_features:
      features_indices = self.indices(feature, len(reads))
      self.example[features_indices] = self.stack_feature(feature)

    self.example = np.expand_dims(self.example, -1)

  def indices(self, feature_name: str, n_reads: int = 0) -> slice:
    """Returns row indices for a given feature.

    Args:
      feature_name: Name of the feature.
      n_reads: Number of reads observed in this region.

    Returns:
      A slice containing the indices corresponding to the feature.
    """
    if n_reads:
      n_rows = min(n_reads, self.max_coverage)
      return slice(
          getattr(self, feature_name),
          getattr(self, feature_name) + n_rows)
    else:
      return slice(
          getattr(self, feature_name),
          getattr(self, feature_name) + self.feature_rows[feature_name])

  def stack_feature(self, feature_name: str) -> np.ndarray:
    """Extract read feature and stack.

    Args:
      feature_name: Name of the feature we want to stack.

    Returns:
      A stacked ndarray of the feature values.
    """
    return np.stack(
        [getattr(x, feature_name) for x in self.reads[:self.max_coverage]])

  @property
  def tensor_height(self) -> int:
    """Returns total rows for tf.Example input."""
    return sum(self.feature_rows.values())

  @property
  def tensor_width(self) -> int:
    """Returns total rows for tf.Example input."""
    return self.example_width

  def to_dict(self) -> dict[str, Any]:
    """Output configuration properties as dict."""
    return {
        # Encode values as strings to prevent downstream aggregation.
        'max_coverage': str(self.max_coverage),
        'example_width': str(self.example_width),
        'tensor_height': str(self.tensor_height),
        'tensor_width': str(self.tensor_width)
    }


def group_nearby_active_positions(
    active_positions: Sequence[int],
    distance_threshold: int,
    positions_to_index: dict[int, int],
) -> list[list[int]]:
  """Group active positions that are within distance_threshold.

  Args:
      active_positions: List of active positions.
      distance_threshold: Distance threshold to use for group.
      positions_to_index: Map of position to index to measure distance.

  Returns:
    A list of lists containing grouped active positions.
  """
  if not active_positions:
    return []

  # We start a stack with first group created.
  active_group = [[active_positions[0]]]

  for current_active_position in active_positions[1:]:
    # get the top element from the stack
    last_active_position = active_group[-1][0]

    distance = positions_to_index[current_active_position] - positions_to_index[
        last_active_position]
    if distance <= distance_threshold:
      active_group[-1].append(current_active_position)
    else:
      active_group.append([current_active_position])

  return active_group


def get_region_length(interval: RegionRecord) -> int:
  """Get length of the region.

  Args:
    interval: A region interval.

  Returns:
    The length of the region.
  """
  return interval.stop - interval.start + 1


def get_tf_examples(
    interval: RegionRecord,
    reads: list[pysam.AlignedSegment],
    reference_sequence: str,
    train_mode: bool,
    ploidy: int,
    bed_interval_regions: list[RegionRecord],
    truths: list[Optional[pysam.AlignedSegment]],
) -> list[Any]:
  """Generates tf examples from reads aligned to a region.

  This is the core process that generates the examples. It works in five steps:

  Step 1: First we calculate how much spacing is needed in this region.
  Step 2: Then we calculate how many active positions are there in this regions.
    - Active positions are positions where non-match rate is above a threshold
    and we want to generate examples of it.
  Step 3: Then we space out the reference and positions in the region.
  Step 4: We encode the reads and references.
  Step 5: We iterate over each active position and create a tf.Example().

  Args:
    interval: A region interval in which examples will be generated.
    reads: Reads alignments in the interval.
    reference_sequence: Reference sequence.
    train_mode: If true then label will be generated.
    ploidy: Number of labels to include, e.g. 1 for haploid, 2 for diploid.
    bed_interval_regions: BED interval regions provided for this region.
    truths: List of truth reads from which to generate labels. Length of truths
      should match ploidy.

  Returns:
    A list of tf.train.Example() for each active position in this region.
  """
  # Step 1: Calculate how much spacing is needed for this region
  if not train_mode:
    spacing_record = PositionalSpacingRecord(interval, reads)
  else:
    # If we are in train mode then truth_read needs to be used for spacing.
    spacing_record = PositionalSpacingRecord(interval, reads + truths)

  # Step 2: Calculate how many active positions are there in this region.
  nonmatch_record = PositionalNonmatchRecord(interval, reads,
                                             reference_sequence)

  # first filter the positions by bed regions
  bed_active_positions = [
      pos for pos in nonmatch_record.active_positions
      if utils_make_images.check_if_position_is_within_regions(
          pos, bed_interval_regions)
  ]

  # Step 3: Space out the reference and the position.
  spaced_ref_record = SpacedReferenceRecord(interval, spacing_record,
                                            reference_sequence)
  # Then filter it by range
  grouped_active_positions = group_nearby_active_positions(
      bed_active_positions,
      encoding.get_active_position_distance_threshold(),
      spaced_ref_record.ref_position_to_index,
  )

  # if there are no active positions then we simply skip this region
  if not grouped_active_positions:
    return []

  window_length = encoding.get_window_length()
  overlap_length = encoding.get_overlap_length()
  # spaced length determines the arrays we generate during encoding.
  # its value has to be minimum the length of the window for proper encoding.
  spaced_length = max(window_length, spacing_record.region_length_with_spacing)

  encoded_reference = EncodedReference(spacing_record, spaced_ref_record,
                                       spaced_length)

  # Optional for taining: if train_mode then encode the truth read.
  # encoded_truth_read = None
  encoded_truths = None
  if train_mode:
    encoded_truths = []
    for t in truths:
      encoded_truths.append(
          EncodedRead(
              interval=interval,
              read=t,
              spaced_reference_record=spaced_ref_record,
              spaced_length=spaced_length,
          )
      )

  all_tf_examples = []
  # we keep track of reads we encode, this way we don't encode
  # same read for each example
  encoded_reads = {}
  # we also create a dictionary to look-up reads by their query_name
  reads_dictionary = {}
  for read in reads:
    reads_dictionary[read.query_name] = read

  # Step 4: Iterate over each active positions and create examples.
  for active_positions in grouped_active_positions:
    # Use the first active position to create examples.
    active_position = active_positions[0]
    # get all the read names that intersect with this position
    overlapping_reads_tree = spacing_record.read_span_intervaltree[
        active_position]

    overlapping_encoded_reads = []
    for overlapping_read_interval in overlapping_reads_tree:
      # overlapping_read_interval.data gives us the query name
      # If we have already encoded the read then simply add it to the list
      if overlapping_read_interval.data in encoded_reads:
        overlapping_encoded_reads.append(
            encoded_reads[overlapping_read_interval.data])
      # otherwise grab the read, encode it and add it to the cache
      else:
        read = reads_dictionary[overlapping_read_interval.data]
        encoded_read = EncodedRead(interval, read, spaced_ref_record,
                                   spaced_length)
        overlapping_encoded_reads.append(encoded_read)
        encoded_reads[read.query_name] = encoded_read

    # Create the example using the reads that are overlapping with this
    # active position
    encoded_example = Example(
        max_coverage=encoding.get_max_coverage_per_haplotype(),
        example_width=spaced_length,
        reads=overlapping_encoded_reads,
        encoded_ref=encoded_reference,
    )
    region_start_index = max(
        0, spaced_ref_record.ref_position_to_index[active_position] -
        overlap_length)
    # When we create reference position encoding we do that with some buffer of
    # 20bp. In this case, encoding the last base makes it difficult. So we don't
    # encode active position that's at the very last base.
    if active_position + 1 in spaced_ref_record.ref_position_to_index:
      # in case the insertion/deletion is very large, we will create multiple
      # examples.
      region_end_index = max(
          region_start_index + window_length,
          spaced_ref_record.ref_position_to_index[active_position + 1] +
          overlap_length)
    else:
      region_end_index = spacing_record.region_length_with_spacing - 1
      region_start_index = max(0, region_end_index - window_length)

    # Once we have the indices, we create examples.
    for index in range(region_start_index, region_end_index, window_length):
      start_index = index
      end_index = index + window_length
      if end_index > spaced_length:
        end_index = spaced_length
        start_index = max(0, end_index - window_length)

      # slice features and create tf example
      example = tf.train.Example()
      features = example.features
      data = encoded_example.example[:, start_index:end_index, :]
      reference_positions = spaced_ref_record.reference_positions[
          start_index:end_index
      ]
      reference_indices = spaced_ref_record.reference_indices[
          start_index:end_index
      ]

      labels = np.zeros((2, window_length), dtype=np.uint8)
      if train_mode:
        for i, t in enumerate(encoded_truths):
          labels[i, :] = t.encoded_bases[start_index:end_index]
      active_positions_np = np.zeros(window_length, dtype=np.int32)
      for i, pos in enumerate(active_positions):
        active_positions_np[i] = pos

      name = ''.join([str(interval.contig), '_', str(active_position)])
      features.feature['name'].bytes_list.value.append(name.encode())
      features.feature['contig'].bytes_list.value.append(
          interval.contig.encode())
      features.feature['active_position'].int64_list.value.extend(
          active_positions_np
      )
      features.feature['reference_positions'].int64_list.value.extend(
          reference_positions)
      features.feature['reference_indices'].int64_list.value.extend(
          reference_indices
      )
      features.feature['encoded_reference'].bytes_list.value.append(
          encoded_reference.encoded_reference[start_index:end_index].tobytes()
      )
      features.feature['shape'].int64_list.value.extend(data.shape)
      features.feature['example'].bytes_list.value.append(data.tobytes())

      if ploidy == 1:
        labels = labels[0, :]
      features.feature['label'].bytes_list.value.append(labels.tobytes())
      # print(features)

      all_tf_examples.append(example)
  return all_tf_examples


@dataclasses.dataclass(frozen=True)
class OptionsForProcess:
  """Names options given to each process.

  This avoids fragility of ordering arguments passed to each process with
  Pool.starmap, which takes a list of arguments instead of named arguments.

  bam_file: Path to an alignment bam file.
  fasta_file: Path to a fasta file.
  truth_to_ref: Path to a truth haplotype alignment file.
  bed_regions_by_contig: A dict of regions to include, keyed by contig name.
      When None, process all regions.
  all_intervals: All genomic intervals that needs processing.
  train_mode: If true, then training mode is on and labels will be generated.
  output_filename: Name of the output file.
  process_id: ID for the process (must be < cpus).
  cpus: Total number of CPUs available for processing.
  ploidy: Number of labels to include, e.g. 1 for haploid assembly polishing,
      2 for diploid variant calling.
  """

  bam_file: str
  fasta_file: str
  truth_to_ref: Optional[str]
  bed_regions_by_contig: Optional[dict[str, Any]]
  all_intervals: list[RegionRecord]
  train_mode: bool
  output_filename: str
  process_id: int
  cpus: int
  ploidy: int


def run_process(options: OptionsForProcess) -> dict[str, Any]:
  """Runs core process from input files to output images.

  Args:
    options: An OptionsForProcess object containing all options that would
      normally be given as individual arguments but due to multiprocessing are
      safer to pass as a single object containing multiple named arguments.

  Returns:
    A dictionary with summary of the run.
  """
  bam_file = options.bam_file
  fasta_file = options.fasta_file
  truth_to_ref = options.truth_to_ref
  bed_regions_by_contig = options.bed_regions_by_contig
  all_intervals = options.all_intervals
  train_mode = options.train_mode
  output_filename = options.output_filename
  process_id = options.process_id
  cpus = options.cpus
  ploidy = options.ploidy

  # Generate the output filename for this process:
  tf_options = tf.io.TFRecordOptions(compression_type='GZIP')
  # Append the suffix "process_id.tfrecords.gz" to the output filename.
  process_filename = f'{output_filename}_{process_id}.tfrecords.gz'
  # Create subdirs if necessary.
  tf.io.gfile.makedirs(os.path.dirname(process_filename))

  run_summaries = {
      'output_filename': process_filename,
      'interval_counter': 0,
      'example_counter': 0,
      'skipped_outside_bed_counter': 0,
      'skipped_too_few_truth_reads_counter': 0,
      'skipped_too_many_truth_reads_counter': 0,
      'skipped_too_many_reads_counter': 0,
  }
  with tf.io.TFRecordWriter(process_filename, tf_options) as tf_writer:
    for interval in utils_make_images.get_process_intervals(
        all_intervals, process_id, cpus):
      logging.info('Process %r: Processing interval %r.', process_id, interval)

      # Get the intersected bed regions.
      if interval.contig in bed_regions_by_contig and bed_regions_by_contig[
          interval.contig] is not None:
        bed_regions = utils_make_images.range_intersect(
            interval, bed_regions_by_contig[interval.contig])
      else:
        bed_regions = [interval]

      # To speed up processing during training image generation,
      # skip generating examples where we don't find any bed regions.
      if not bed_regions:
        run_summaries['skipped_outside_bed_counter'] += 1
        continue

      truths = None
      # If in training mode, get truth reads.
      if train_mode:
        # Get truth haplotype alignment for labeling.
        truth_reads = utils_make_images.filter_reads(
            utils_make_images.get_reads_from_bam(truth_to_ref, interval),
            encoding.get_truth_min_mapping_quality(),
            allow_supplementary=True,
        )

        # Skip intervals where the number of truth sequences does not equal the
        # expected number, which is the ploidy.
        if len(truth_reads) > ploidy:
          run_summaries['skipped_too_many_truth_reads_counter'] += 1
          continue
        elif len(truth_reads) < ploidy:
          run_summaries['skipped_too_few_truth_reads_counter'] += 1
          continue

        truths = truth_reads[0 : ploidy + 1]

        # Set truth reads' name so we can avoid using these reads during
        # mismatch calculation. We don't expect any read from a sequencer to
        # have truth_read as sequence name.
        for t in truths:
          t.query_name = 'truth_read'

      # Get all reads.
      reads = utils_make_images.get_reads_from_bam(bam_file, interval)
      logging.info('Process %r: Total reads found %r', process_id, len(reads))
      # If a region has more than _MAX_ALLOWED_READS_IN_REGION reads then
      # most likely it's a collpased or false duplication. We don't want to
      # polish those regions so report and skip.
      if len(reads) > encoding.get_max_allowed_reads_in_region():
        logging.info(
            'Process %r: Too many reads (%r) in region %r.',
            process_id,
            len(reads),
            interval,
        )
        run_summaries['skipped_too_many_reads_counter'] += 1
        continue
      # Filter reads.
      reads = utils_make_images.filter_reads(
          reads, encoding.get_min_mapping_quality()
      )
      reference_sequence = utils_make_images.get_reference_sequence_from_fasta(
          fasta_file, interval, encoding.get_padding_for_reference_sequence()
      )

      total_examples = 0
      if reads:
        all_tf_examples = get_tf_examples(
            interval=interval,
            reads=reads,
            reference_sequence=reference_sequence,
            train_mode=train_mode,
            ploidy=ploidy,
            bed_interval_regions=bed_regions,
            truths=truths,
        )

        produced_examples = []
        for example in all_tf_examples:
          produced_examples.append(example.SerializeToString())
          total_examples += 1
        if produced_examples:
          utils_make_images.write_tf_record(produced_examples, tf_writer)
      else:
        logging.info('Process %r: No reads found in %r.', process_id, interval)

      run_summaries['example_counter'] += total_examples
      run_summaries['interval_counter'] += 1

  return run_summaries
