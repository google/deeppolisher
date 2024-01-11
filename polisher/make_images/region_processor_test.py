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
"""Tests for region_processor."""

import dataclasses
import itertools
from typing import Any, List, Tuple

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pysam
import tensorflow as tf

from polisher.make_images import encoding
from polisher.make_images import region_processor
from polisher.make_images import test_utils
from polisher.make_images import utils_make_images


@dataclasses.dataclass
class ReadAttributes:
  """Represents attributes of a read used for test.

  Attributes:
    bam_file: Path to an alignment bam file to get optional read attributes.
    query_name: Name of the read.
    query_sequence: Nucleotide sequence of read.
    contig: Contig name where read is aligned.
    start: Position where read is aligned.
    base_qualities: Read base qualities.
    mapping_quality: Mapping quality of the read.
    cigartuples: Cigartuple describing the alignment of the read.
    tags: Auxiliary tags for the reads.
  """
  bam_file: str
  query_name: str
  query_sequence: str
  contig: str
  start: int
  base_qualities: List[int]
  mapping_quality: int
  cigartuples: List[Tuple[int, int]]
  tags: List[Tuple[str, Any]]


class Test(parameterized.TestCase):
  """Tests for region processor component of the polisher."""

  def set_read_attributes(self, read):
    """Set attributes of reads."""
    read.is_duplicate = False
    read.is_qcfail = False
    read.is_secondary = False
    read.is_unmapped = False
    read.is_supplementary = False
    read.mapping_quality = 60

  def get_pysam_read_object(self, read_attributes):
    """Creates a pysam read object given attributes."""
    bam_file = test_utils.polisher_testdata(read_attributes.bam_file)
    pysam_bam = pysam.AlignmentFile(bam_file)
    read = pysam.AlignedSegment(pysam_bam.header)
    self.set_read_attributes(read)
    read.query_name = read_attributes.query_name
    read.query_sequence = read_attributes.query_sequence
    read.reference_name = read_attributes.contig
    read.reference_start = read_attributes.start
    read.query_qualities = read_attributes.base_qualities
    read.cigartuples = read_attributes.cigartuples
    read.mapping_quality = read_attributes.mapping_quality
    if read_attributes.tags:
      for tag, value in read_attributes.tags:
        read.set_tag(tag, value)
    return read

  @parameterized.parameters(
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 11,
              "positional_max_observed_insert": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "observed_nonmatches": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "observed_nonmatch_frequencies": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
              ],
              "active_positions": [],
              "reference_positions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              "reference_indices": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 3,
                  4: 4,
                  5: 5,
                  6: 6,
                  7: 7,
                  8: 8,
                  9: 9,
                  10: 10
              },
              "spaced_reference_sequence": "ACGTACGTACA"
          },
          message="Test 1: Simple test one read."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTATACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 9,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5), (pysam.CDEL, 2),
                               (pysam.CMATCH, 4)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 11,
              "positional_max_observed_insert": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "observed_nonmatches": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              "observed_nonmatch_frequencies": [
                  0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0
              ],
              "active_positions": [],
              "reference_positions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              "reference_indices": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 3,
                  4: 4,
                  5: 5,
                  6: 6,
                  7: 7,
                  8: 8,
                  9: 9,
                  10: 10
              },
              "spaced_reference_sequence": "ACGTACGTACA"
          },
          message="Test 2: Simple test with one del, no padding required."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTAAACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 13,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5), (pysam.CINS, 2),
                               (pysam.CMATCH, 6)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 13,
              "positional_max_observed_insert": [
                  0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "observed_nonmatches": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              "observed_nonmatch_frequencies": [
                  0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0
              ],
              "active_positions": [],
              "reference_positions": [0, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10],
              "reference_indices": [0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 3,
                  4: 4,
                  5: 7,
                  6: 8,
                  7: 9,
                  8: 10,
                  9: 11,
                  10: 12
              },
              "spaced_reference_sequence": "ACGTA**CGTACA"
          },
          message="Test 3: Test with one insertion, padding needed."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGACTAACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 14,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 3), (pysam.CINS, 2),
                               (pysam.CMATCH, 2), (pysam.CINS, 1),
                               (pysam.CMATCH, 6)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 14,
              "positional_max_observed_insert": [
                  0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "observed_nonmatches": [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              "observed_nonmatch_frequencies": [
                  0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0
              ],
              "active_positions": [],
              "reference_positions": [
                  0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10
              ],
              "reference_indices": [0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 5,
                  4: 6,
                  5: 8,
                  6: 9,
                  7: 10,
                  8: 11,
                  9: 12,
                  10: 13
              },
              "spaced_reference_sequence": "ACG**TA*CGTACA"
          },
          message="Test 4: Test with two insertions, padding needed."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="AGACTAACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 13,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 1), (pysam.CDEL, 1),
                               (pysam.CMATCH, 1), (pysam.CINS, 2),
                               (pysam.CMATCH, 2), (pysam.CINS, 1),
                               (pysam.CMATCH, 6)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 14,
              "positional_max_observed_insert": [
                  0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "observed_nonmatches": [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              "observed_nonmatch_frequencies": [
                  0, 1.0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0
              ],
              "active_positions": [],
              "reference_positions": [
                  0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10
              ],
              "reference_indices": [0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 5,
                  4: 6,
                  5: 8,
                  6: 9,
                  7: 10,
                  8: 11,
                  9: 12,
                  10: 13
              },
              "spaced_reference_sequence": "ACG**TA*CGTACA"
          },
          message="Test 5: Test with two ins and a del, padding needed."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="AGACTAACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 13,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 1), (pysam.CDEL, 1),
                               (pysam.CMATCH, 1), (pysam.CINS, 2),
                               (pysam.CMATCH, 2), (pysam.CINS, 1),
                               (pysam.CMATCH, 6)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="AGACTAACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 13,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 1), (pysam.CDEL, 1),
                               (pysam.CMATCH, 1), (pysam.CINS, 2),
                               (pysam.CMATCH, 2), (pysam.CINS, 1),
                               (pysam.CMATCH, 6)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 14,
              "positional_max_observed_insert": [
                  0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
              "observed_nonmatches": [0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0],
              "observed_nonmatch_frequencies": [
                  0, 1.0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0
              ],
              "active_positions": [1, 2, 4],
              "reference_positions": [
                  0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10
              ],
              "reference_indices": [0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 5,
                  4: 6,
                  5: 8,
                  6: 9,
                  7: 10,
                  8: 11,
                  9: 12,
                  10: 13
              },
              "spaced_reference_sequence": "ACG**TA*CGTACA"
          },
          message="Test 6: Test with two reads for active positions."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="AGACTAACGTATA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 13,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 1), (pysam.CDEL, 1),
                               (pysam.CMATCH, 1), (pysam.CINS, 2),
                               (pysam.CMATCH, 2), (pysam.CINS, 1),
                               (pysam.CMATCH, 6)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="AGACTAACGTACAGTGTGT",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 19,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 1), (pysam.CDEL, 1),
                               (pysam.CMATCH, 1), (pysam.CINS, 2),
                               (pysam.CMATCH, 2), (pysam.CINS, 1),
                               (pysam.CMATCH, 12)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 14,
              "positional_max_observed_insert": [
                  0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
              "observed_nonmatches": [0, 2, 2, 0, 2, 0, 0, 0, 0, 1, 0],
              "observed_nonmatch_frequencies": [
                  0, 1.0, 1.0, 0, 1.0, 0, 0, 0, 0, 0.5, 0
              ],
              "active_positions": [1, 2, 4],
              "reference_positions": [
                  0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10
              ],
              "reference_indices": [0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 5,
                  4: 6,
                  5: 8,
                  6: 9,
                  7: 10,
                  8: 11,
                  9: 12,
                  10: 13
              },
              "spaced_reference_sequence": "ACG**TA*CGTACA"
          },
          message="Test 7: Test two reads for active positions with mismatch."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 21),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 77,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11), (pysam.CINS, 55),
                               (pysam.CMATCH, 11)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACAACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 22,
              "positional_max_observed_insert": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "observed_depth": [
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1
              ],
              "observed_nonmatches": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "observed_nonmatch_frequencies": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "active_positions": [],
              "reference_positions": [
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21
              ],
              "reference_indices": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 3,
                  4: 4,
                  5: 5,
                  6: 6,
                  7: 7,
                  8: 8,
                  9: 9,
                  10: 10,
                  11: 11,
                  12: 12,
                  13: 13,
                  14: 14,
                  15: 15,
                  16: 16,
                  17: 17,
                  18: 18,
                  19: 19,
                  20: 20,
                  21: 21,
              },
              "spaced_reference_sequence": "ACGTACGTACAACGTACGTACA"
          },
          message="Test 8: Large insertion that should be ignored."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 21),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACAACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 22,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11), (pysam.CDEL, 55),
                               (pysam.CMATCH, 11)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 22,
              "positional_max_observed_insert": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "observed_depth": [
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1
              ],
              "observed_nonmatches": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "observed_nonmatch_frequencies": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "active_positions": [],
              "reference_positions": [
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21
              ],
              "reference_indices": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0
              ],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 3,
                  4: 4,
                  5: 5,
                  6: 6,
                  7: 7,
                  8: 8,
                  9: 9,
                  10: 10,
                  11: 11,
                  12: 12,
                  13: 13,
                  14: 14,
                  15: 15,
                  16: 16,
                  17: 17,
                  18: 18,
                  19: 19,
                  20: 20,
                  21: 21,
              },
              "spaced_reference_sequence": "ACGTACGTACAACGTACGTACA"
          },
          message="Test 8: Large deletion that should be ignored."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          reads=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="truth_read",
                  query_sequence="AGACTAACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 13,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 1), (pysam.CDEL, 1),
                               (pysam.CMATCH, 1), (pysam.CINS, 2),
                               (pysam.CMATCH, 2), (pysam.CINS, 1),
                               (pysam.CMATCH, 6)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTACGTACA",
                  contig="chr20",
                  start=0,
                  base_qualities=[60] * 11,
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 11)],
                  tags=[("HP", 1)])
          ],
          reference_sequence="ACGTACGTACA",
          expected_values={
              "region_length_with_spacing": 14,
              "positional_max_observed_insert": [
                  0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0
              ],
              "observed_depth": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              "observed_nonmatches": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "observed_nonmatch_frequencies": [
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
              ],
              "active_positions": [],
              "reference_positions": [
                  0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10
              ],
              "reference_indices": [0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              "ref_position_to_index": {
                  0: 0,
                  1: 1,
                  2: 2,
                  3: 5,
                  4: 6,
                  5: 8,
                  6: 9,
                  7: 10,
                  8: 11,
                  9: 12,
                  10: 13
              },
              "spaced_reference_sequence": "ACG**TA*CGTACA"
          },
          message="Test 7: Test with truth_read encoding, spacing performed but no active positions."
      ),
  )
  @flagsaver.flagsaver
  def test_positional_spacing(self, interval, reads, reference_sequence,
                              expected_values, message):
    """Test dataclasses that handle positional spacing.

    Dataclasses tested in this module are:
      PositionalSpacingRecord
      PositionalNonmatchRecord
      SpacedReferenceRecord

    Args:
      interval: A region interval in which examples will be generated.
      reads: Reads alignments in the interval.
      reference_sequence: Reference sequence.
      expected_values: Values we expect the methods to return.
      message: Message to print for the test.
    """
    all_reads = [
        self.get_pysam_read_object(read_attr)
        for read_attr in reads
        if read_attr.query_name != "truth_read"
    ]
    truth_reads = [
        self.get_pysam_read_object(read_attr)
        for read_attr in reads
        if read_attr.query_name == "truth_read"
    ]

    spacing_record = region_processor.PositionalSpacingRecord(
        interval, all_reads + truth_reads)

    interval_names = [
        read_interval.data
        for read_interval in spacing_record.read_span_intervaltree.items()
    ]

    # Make sure that truth reads didn't end up in the interval tree.
    for read in truth_reads:
      self.assertNotIn(
          read.query_name,
          interval_names,
          msg=": ".join([message, " truth read intervaltree", read.query_name]))

    # And all the other reads are in the interval tree.
    for read in all_reads:
      self.assertIn(
          read.query_name,
          interval_names,
          msg=": ".join([message, " read in intervaltree", read.query_name]))

    nonmatch_record = region_processor.PositionalNonmatchRecord(
        interval, all_reads, reference_sequence)

    spaced_ref_record = region_processor.SpacedReferenceRecord(
        interval, spacing_record, reference_sequence)

    ## spacing_record attributes
    self.assertListEqual(
        expected_values["positional_max_observed_insert"],
        spacing_record.max_observed_insert.tolist(),
        msg=": ".join([message, "positional_max_observed_insert"]))
    self.assertEqual(
        expected_values["region_length_with_spacing"],
        spacing_record.region_length_with_spacing,
        msg=": ".join([message, "region_length_with_spacing"]))

    ## nonmatch_record attributes
    self.assertListEqual(
        expected_values["observed_depth"],
        nonmatch_record.observed_depth.tolist(),
        msg=": ".join([message, "observed_depth"]))
    self.assertListEqual(
        expected_values["observed_nonmatches"],
        nonmatch_record.observed_nonmatches.tolist(),
        msg=": ".join([message, "observed_nonmatches"]))
    self.assertListEqual(
        expected_values["observed_nonmatch_frequencies"],
        nonmatch_record.observed_nonmatch_frequencies.tolist(),
        msg=": ".join([message, "observed_nonmatch_frequencies"]))
    self.assertListEqual(
        expected_values["active_positions"],
        nonmatch_record.active_positions,
        msg=": ".join([message, "active_positions"]))

    ## spaced_ref_record attributes
    self.assertListEqual(
        expected_values["reference_positions"],
        spaced_ref_record.reference_positions.tolist(),
        msg=": ".join([message, "reference_positions"]))
    self.assertListEqual(
        expected_values["reference_indices"],
        spaced_ref_record.reference_indices.tolist(),
        msg=": ".join([message, "reference_indices"]))
    self.assertDictEqual(
        expected_values["ref_position_to_index"],
        spaced_ref_record.ref_position_to_index,
        msg=": ".join([message, "ref_position_to_index"]))
    self.assertEqual(
        expected_values["spaced_reference_sequence"],
        spaced_ref_record.spaced_reference_sequence,
        msg=": ".join([message, "spaced_reference_sequence"]))

  @parameterized.parameters(
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          read_attrs=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACTTA",
                  contig="chr20",
                  start=0,
                  base_qualities=[1, 2, 3, 4, 5],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACTTA",
                  contig="chr20",
                  start=0,
                  base_qualities=[1, 2, 3, 4, 5],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_3",
                  query_sequence="ACGTACG",
                  contig="chr20",
                  start=5,
                  base_qualities=[5, 6, 7, 8, 9, 10, 11],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 7)],
                  tags=[("HP", 1)])
          ],
          ref_seq="ACGTACGTACG",
          expected_values={
              "expected_start_index": 0,
              "expected_end_index": 100,
              "expected_active_position": [2],
              "example_reads": ["read_1", "read_2"]
          },
          message="Test 1: Create example with one read outside right interval."
      ),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          read_attrs=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACGTA",
                  contig="chr20",
                  start=0,
                  base_qualities=[1, 2, 3, 4, 5],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACGTA",
                  contig="chr20",
                  start=0,
                  base_qualities=[1, 2, 3, 4, 5],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_3",
                  query_sequence="TGTACGT",
                  contig="chr20",
                  start=5,
                  base_qualities=[5, 6, 7, 8, 9, 10, 11],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 7)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_4",
                  query_sequence="TGTACGT",
                  contig="chr20",
                  start=5,
                  base_qualities=[5, 6, 7, 8, 9, 10, 11],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 7)],
                  tags=[("HP", 1)])
          ],
          ref_seq="ACGTACGTACG",
          expected_values={
              "expected_start_index": 0,
              "expected_end_index": 100,
              "expected_active_position": [5],
              "example_reads": ["read_3", "read_4"]
          },
          message="Test 2: Create example with one read outside left interval."
      ),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          read_attrs=[
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="truth_read",
                  query_sequence="ACTTTTA",
                  contig="chr20",
                  start=0,
                  base_qualities=[1, 2, 2, 2, 3, 4, 5],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 2), (pysam.CMATCH, 2),
                               (pysam.CINS, 3)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_1",
                  query_sequence="ACTTA",
                  contig="chr20",
                  start=0,
                  base_qualities=[1, 2, 3, 4, 5],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_2",
                  query_sequence="ACTTA",
                  contig="chr20",
                  start=0,
                  base_qualities=[1, 2, 3, 4, 5],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 5)],
                  tags=[("HP", 1)]),
              ReadAttributes(
                  bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
                  query_name="read_3",
                  query_sequence="ACGTACG",
                  contig="chr20",
                  start=5,
                  base_qualities=[5, 6, 7, 8, 9, 10, 11],
                  mapping_quality=60,
                  cigartuples=[(pysam.CMATCH, 7)],
                  tags=[("HP", 1)])
          ],
          ref_seq="ACGTACGTACG",
          expected_values={
              "expected_start_index": 0,
              "expected_end_index": 100,
              "expected_active_position": [2],
              "example_reads": ["read_1", "read_2"]
          },
          message="Test 3: Create example with truth_read."),
  )
  @flagsaver.flagsaver
  def test_get_tf_examples(self, interval, read_attrs, ref_seq, expected_values,
                           message):
    """Test get_tf_examples class.

    Args:
      interval: A region interval in which examples will be generated.
      read_attrs: Attributes of a read.
      ref_seq: Reference sequence.
      expected_values: Values we expect the methods to return.
      message: Message to print for the test.
    """
    # first convert the reads to pysam reads
    reads = [
        self.get_pysam_read_object(read_attr)
        for read_attr in read_attrs
        if read_attr.query_name != "truth_read"
    ]
    truth_reads = [
        self.get_pysam_read_object(read_attr)
        for read_attr in read_attrs
        if read_attr.query_name == "truth_read"
    ]

    train_mode = False
    if truth_reads:
      self.assertLen(
          truth_reads, 1, msg="test_get_tf_examples: truth_reads length check.")
      train_mode = True

    # Now we create the expected example, but it requires calling some
    # methods that are explicitly tested in other methods.

    # spacing record is tested separately so we don't test it here.
    spacing_record = region_processor.PositionalSpacingRecord(
        interval, reads + truth_reads
    )

    # speced ref record is tested separately so we don't test it here.
    spaced_ref_record = region_processor.SpacedReferenceRecord(
        interval, spacing_record, ref_seq
    )

    window_length = encoding.get_window_length()
    spaced_length = max(
        window_length, spacing_record.region_length_with_spacing
    )

    # Convert the expected active position to a list.
    expected_active_position_list = list(
        expected_values["expected_active_position"]
    )
    expected_active_position_list.extend(
        [0] * (window_length - len(expected_values["expected_active_position"]))
    )
    expected_values["expected_active_position"] = expected_active_position_list

    # encoded reference is tested separately.
    encoded_reference = region_processor.EncodedReference(
        spacing_record, spaced_ref_record, spaced_length
    )

    # we manually select the reads that overlaps with the active position
    # this eventually tests the example we create manually with the one
    # that is returned from the function.
    overlapping_encoded_reads = []
    for read in reads:
      if read.query_name in expected_values["example_reads"]:
        encoded_read = region_processor.EncodedRead(interval, read,
                                                    spaced_ref_record,
                                                    spaced_length)
        overlapping_encoded_reads.append(encoded_read)

    # create the example and get the expected return
    expected_encoded_example = region_processor.Example(
        max_coverage=encoding.get_max_coverage_per_haplotype(),
        example_width=spaced_length,
        reads=overlapping_encoded_reads,
        encoded_ref=encoded_reference,
    )
    expected_data = expected_encoded_example.example[:, expected_values[
        "expected_start_index"]:expected_values["expected_end_index"], :]

    # create the expected train example
    expected_example = tf.train.Example()
    expected_example.features.feature["example"].bytes_list.value.append(
        expected_data.tobytes())
    expected_example.features.feature[
        "active_position"].int64_list.value.extend(
            expected_values["expected_active_position"])

    # We get the example by calling the main method which we want to test
    all_tf_examples = region_processor.get_tf_examples(
        interval=interval,
        reads=reads,
        reference_sequence=ref_seq,
        train_mode=train_mode,
        ploidy=1,
        truths=truth_reads,
        bed_interval_regions=[interval],
    )
    # make sure we only have one example which is expected.
    self.assertLen(all_tf_examples, 1)
    returned_tf_example = all_tf_examples[0]
    returned_saved_example = returned_tf_example.features.feature[
        "example"].bytes_list
    returned_active_positions = returned_tf_example.features.feature[
        "active_position"].int64_list
    # get the expected example we created
    expected_saved_example = expected_example.features.feature[
        "example"].bytes_list
    expected_active_position = expected_example.features.feature[
        "active_position"].int64_list
    # compare
    self.assertEqual(
        expected_saved_example, returned_saved_example, msg=message)
    self.assertEqual(
        expected_active_position, returned_active_positions, msg=message)

  @parameterized.parameters(
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="ACGTACGTACA",
              contig="chr20",
              start=0,
              base_qualities=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              mapping_quality=60,
              cigartuples=[(pysam.CMATCH, 11)],
              tags=[("HP", 1)]),
          reference_sequence="ACGTACGTACA",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACGTACGTACA")
              ],
              "encoded_base_qualities": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              "encoded_mapping_quality": [60] * 11,
              "encoded_match_mismatch":
                  [encoding._MATCH_MISMATCH_ENCODINGS["M"]] * 11,
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACGTACGTACA")
              ],
          },
          message="Test 1: Simple encoding test one read."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 3),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="ACCGT",
              contig="chr20",
              start=0,
              base_qualities=[121, 2, 3, 4, 5],
              mapping_quality=121,
              cigartuples=[(pysam.CMATCH, 2), (pysam.CINS, 1),
                           (pysam.CMATCH, 2)],
              tags=[("HP", 1)]),
          reference_sequence="ACGT",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACCGT")
              ],
              "encoded_base_qualities": [
                  encoding._BASE_QUALITY_CAP, 2, 3, 4, 5
              ],
              "encoded_mapping_quality": [encoding._MAPPING_QUALITY_CAP] * 5,
              "encoded_match_mismatch": [
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
              ],
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("AC*GT")
              ],
          },
          message="Test 2: Encoding test with insert."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 3),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="ACCGT",
              contig="chr20",
              start=0,
              base_qualities=[121, 2, 3, 4, 5],
              mapping_quality=121,
              cigartuples=[(pysam.CMATCH, 2), (pysam.CINS, 1),
                           (pysam.CMATCH, 2)],
              tags=[("HP", 1)]),
          reference_sequence="ACGC",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACCGT")
              ],
              "encoded_base_qualities": [
                  encoding._BASE_QUALITY_CAP, 2, 3, 4, 5
              ],
              "encoded_mapping_quality": [encoding._MAPPING_QUALITY_CAP] * 5,
              "encoded_match_mismatch": [
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
              ],
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("AC*GC")
              ],
          },
          message="Test 3: Encoding test with insert and one mismatch."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 3),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="ACCT",
              contig="chr20",
              start=0,
              base_qualities=[121, 2, 3, 4],
              mapping_quality=121,
              cigartuples=[(pysam.CMATCH, 2), (pysam.CINS, 1),
                           (pysam.CMATCH, 1), (pysam.CDEL, 1)],
              tags=[("HP", 1)]),
          reference_sequence="ACGC",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACCT*")
              ],
              "encoded_base_qualities": [
                  encoding._BASE_QUALITY_CAP, 2, 3, 4, 0
              ],
              "encoded_mapping_quality": [encoding._MAPPING_QUALITY_CAP] * 5,
              "encoded_match_mismatch": [
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
              ],
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("AC*GC")
              ],
          },
          message="Test 4: Encoding test with insert, delete and mismatch."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 3),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="CCACCT",
              contig="chr20",
              start=0,
              base_qualities=[2, 2, 121, 2, 3, 4],
              mapping_quality=121,
              cigartuples=[(pysam.CINS, 2), (pysam.CMATCH, 2), (pysam.CINS, 1),
                           (pysam.CMATCH, 1), (pysam.CDEL, 1)],
              tags=[("HP", 1)]),
          reference_sequence="ACGC",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACCT*")
              ],
              "encoded_base_qualities": [
                  encoding._BASE_QUALITY_CAP, 2, 3, 4, 0
              ],
              "encoded_mapping_quality": [encoding._MAPPING_QUALITY_CAP] * 5,
              "encoded_match_mismatch": [
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
              ],
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("AC*GC")
              ],
          },
          message="Test 5: Encoding test with leading insert."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 3),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="ANGT",
              contig="chr20",
              start=0,
              base_qualities=[1, 2, 3, 4],
              mapping_quality=60,
              cigartuples=[(pysam.CMATCH, 11)],
              tags=[("HP", 0)]),
          reference_sequence="ACGTACGTACA",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS["A"], 0,
                  encoding._BASE_ENCODINGS["G"], encoding._BASE_ENCODINGS["T"]
              ],
              "encoded_base_qualities": [1, 2, 3, 4],
              "encoded_mapping_quality": [60, 60, 60, 60],
              "encoded_match_mismatch": [
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["X"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"],
                  encoding._MATCH_MISMATCH_ENCODINGS["M"]
              ],
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACGT")
              ],
          },
          message="Test 6: Encoding test with haplotype 0 and N bases."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 21),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="ACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACA",
              contig="chr20",
              start=0,
              base_qualities=[60] * 77,
              mapping_quality=60,
              cigartuples=[(pysam.CMATCH, 11), (pysam.CINS, 55),
                           (pysam.CMATCH, 11)],
              tags=[("HP", 1)]),
          reference_sequence="ACGTACGTACAACGTACGTACA",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACGTACGTACAACGTACGTACA")
              ],
              "encoded_base_qualities": [60] * 22,
              "encoded_mapping_quality": [60] * 22,
              "encoded_match_mismatch":
                  [encoding._MATCH_MISMATCH_ENCODINGS["M"]] * 22,
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACGTACGTACAACGTACGTACA")
              ],
          },
          message="Test 7: Large insertion that should be ignored."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 21),
          read_attr=ReadAttributes(
              bam_file="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
              query_name="read_1",
              query_sequence="ACGTACGTACAACGTACGTACA",
              contig="chr20",
              start=0,
              base_qualities=[60] * 22,
              mapping_quality=60,
              cigartuples=[(pysam.CMATCH, 11), (pysam.CDEL, 55),
                           (pysam.CMATCH, 11)],
              tags=[("HP", 1)]),
          reference_sequence="ACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACAACGTACGTACA",
          expected_values={
              "encoded_bases": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACGTACGTACA***********")
              ],
              "encoded_base_qualities": [60] * 11 + [0] * 11,
              "encoded_mapping_quality": [60] * 22,
              "encoded_match_mismatch":
                  [encoding._MATCH_MISMATCH_ENCODINGS["M"]] * 11 +
                  [encoding._MATCH_MISMATCH_ENCODINGS["X"]] * 11,
              "encoded_reference": [
                  encoding._BASE_ENCODINGS[base.upper()]
                  for base in list("ACGTACGTACAACGTACGTACA")
              ],
          },
          message="Test 8: Large deletion has no affect during encoding."),
  )
  @flagsaver.flagsaver
  def test_encoding_dataclasses(self, interval, read_attr, reference_sequence,
                                expected_values, message):
    """Test dataclasses that handle positional spacing.

    Dataclasses tested in this module are:
      EncodedRead
      EncodedReference
      Example

    Args:
      interval: A region interval in which examples will be generated.
      read_attr: Attributes of a read.
      reference_sequence: Reference sequence.
      expected_values: Values we expect the methods to return.
      message: Message to print for the test.
    """
    read = self.get_pysam_read_object(read_attr)

    spacing_record = region_processor.PositionalSpacingRecord(interval, [read])

    spaced_ref_record = region_processor.SpacedReferenceRecord(
        interval, spacing_record, reference_sequence)

    spaced_length = spacing_record.region_length_with_spacing
    encoded_read = region_processor.EncodedRead(interval, read,
                                                spaced_ref_record,
                                                spaced_length)
    encoded_reference = region_processor.EncodedReference(
        spacing_record, spaced_ref_record, spaced_length)

    self.assertEqual(
        encoded_read.encoded_bases.tolist(),
        expected_values["encoded_bases"],
        msg=": ".join([message, "encoded_bases"]))

    self.assertEqual(
        encoded_read.encoded_base_qualities.tolist(),
        expected_values["encoded_base_qualities"],
        msg=": ".join([message, "encoded_base_qualities"]))

    self.assertEqual(
        encoded_read.encoded_base_qualities.tolist(),
        expected_values["encoded_base_qualities"],
        msg=": ".join([message, "encoded_base_qualities"]))

    self.assertEqual(
        encoded_read.encoded_match_mismatch.tolist(),
        expected_values["encoded_match_mismatch"],
        msg=": ".join([message, "encoded_match_mismatch"]))

    self.assertEqual(
        encoded_reference.encoded_reference.tolist(),
        expected_values["encoded_reference"],
        msg=": ".join([message, "encoded_reference"]))

    example = region_processor.Example(
        max_coverage=1,
        example_width=spaced_length,
        reads=[encoded_read],
        encoded_ref=encoded_reference)

    self.assertLen(
        example.feature_rows.items(),
        example.tensor_height,
        msg=": ".join([message, "tensor_height"]))
    self.assertEqual(
        example.tensor_width,
        spaced_length,
        msg=": ".join([message, "tensor_width"]))

    expected_example_dict = {
        "max_coverage": str(1),
        "example_width": str(spaced_length),
        "tensor_height": str(len(example.feature_rows.items())),
        "tensor_width": str(spaced_length)
    }
    example_dict = example.to_dict()
    self.assertDictEqual(
        example_dict,
        expected_example_dict,
        msg=": ".join([message, "example_dict"]))

    self.assertListEqual(
        list(
            itertools.chain(
                *example.example[example.indices("encoded_bases")].tolist())),
        [[val] for val in expected_values["encoded_bases"]],
        msg=": ".join([message, "example_encoded_bases"]))

    self.assertListEqual(
        list(
            itertools.chain(*example.example[example.indices(
                "encoded_match_mismatch")].tolist())),
        [[val] for val in expected_values["encoded_match_mismatch"]],
        msg=": ".join([message, "example_encoded_match_mismatch"]))

    self.assertListEqual(
        list(
            itertools.chain(*example.example[example.indices(
                "encoded_base_qualities")].tolist())),
        [[val] for val in expected_values["encoded_base_qualities"]],
        msg=": ".join([message, "example_encoded_base_qualities"]))

    self.assertListEqual(
        list(
            itertools.chain(*example.example[example.indices(
                "encoded_mapping_quality")].tolist())),
        [[val] for val in expected_values["encoded_mapping_quality"]],
        msg=": ".join([message, "example_encoded_mapping_quality"]))

  @parameterized.parameters(
      dict(
          interval=utils_make_images.RegionRecord("chr20", 0, 10),
          expected_length=11,
          message="Test 1: Simple region length test."),
      dict(
          interval=utils_make_images.RegionRecord("chr20", 10, 20),
          expected_length=11,
          message="Test 2: Simple region length test."))
  @flagsaver.flagsaver
  def test_get_region_length(self, interval, expected_length, message):
    """Test get_region_length method."""
    region_length = region_processor.get_region_length(interval)
    self.assertEqual(region_length, expected_length, msg=message)

  def load_tf_example(self, element):
    data = {
        "name":
            tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "contig":
            tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "active_position":
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "reference_positions":
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "reference_indices":
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "shape":
            tf.io.FixedLenFeature([3], tf.int64),
        "example":
            tf.io.FixedLenFeature([], tf.string),
        "label":
            tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(serialized=element, features=data)
    name = content["name"]
    contig = content["contig"]
    active_pos = content["active_position"]
    ref_positions = content["reference_positions"]
    ref_indices = content["reference_indices"]

    example = tf.io.decode_raw(content["example"], tf.uint8)
    example = tf.reshape(example, content["shape"])
    label = tf.io.decode_raw(content["label"], tf.uint8)
    return name, contig, active_pos, ref_positions, ref_indices, example, label

  @parameterized.parameters(
      dict(
          bam="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
          fasta="GRCh38_chr20_0_200000.fa",
          region="chr20:85625-85832",
          bed="HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed",
          truth_bam="HG002_chr20_0_200000_GIABv42_2_GRCh38.hap1.bam",
          train_mode=True,
          expected_total_examples=1,
          message="Test 1: chr20:85625-85832 one example."),
      dict(
          bam="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
          fasta="GRCh38_chr20_0_200000.fa",
          region="chr20:101754-101852",
          bed="HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed",
          truth_bam="HG002_chr20_0_200000_GIABv42_2_GRCh38.hap1.bam",
          train_mode=True,
          expected_total_examples=0,
          message="Test 2: chr20:101754-101852 one example."),
      dict(
          bam="HG002_chr20_0_200000_hifi_2_GRCh38.bam",
          fasta="GRCh38_chr20_0_200000.fa",
          region="chr20:0-100",
          bed="HG002_GRCh38_1_22_v4.2.1_benchmark.chr20_0_200000.bed",
          truth_bam="HG002_chr20_0_200000_GIABv42_2_GRCh38.hap1.bam",
          train_mode=True,
          expected_total_examples=0,
          message="Test 3: chr20:0-100 no examples, no reads aligned."),
      # This is an acrocentric region that is collapsed due to missing sequence
      # in GRCh38 which results in extremely high coverage.
      # This test will make sure that we are skipping this region.
      dict(
          bam="HG002_chr16_46380000_46385000_hifi_2_GRCh38.bam",
          fasta="GRCh38_chr16_0_46390000.fa.gz",
          region="chr16:46380000-46385000",
          bed=None,
          truth_bam=None,
          train_mode=False,
          expected_total_examples=0,
          message="Test 5: chr16:46380000-46385000 acrocentric, skip region."))
  @flagsaver.flagsaver
  def test_region_processor(self, bam, fasta, region, bed, truth_bam,
                            train_mode, expected_total_examples, message):
    """Test region process with real data."""
    bam_file = test_utils.polisher_testdata(bam)
    fasta_file = test_utils.polisher_testdata(fasta)
    high_conf_bed_file = test_utils.polisher_testdata(bed) if bed else None
    truth_hap_bam = test_utils.polisher_testdata(
        truth_bam) if truth_bam else None

    filename = "make_examples@process.tfrecord.gz"
    interval_size = 1000

    all_intervals = utils_make_images.get_contig_regions(
        bam_file, fasta_file, region, interval_size)

    bed_regions = utils_make_images.read_bed(
        high_conf_bed_file) if bed else dict()

    output_filename = test_utils.test_tmpfile(filename)
    options = region_processor.OptionsForProcess(
        bam_file=bam_file,
        fasta_file=fasta_file,
        truth_to_ref=truth_hap_bam,
        bed_regions_by_contig=bed_regions,
        all_intervals=all_intervals,
        train_mode=train_mode,
        output_filename=output_filename,
        process_id=0,
        cpus=1,
        ploidy=1,
    )
    run_summary = region_processor.run_process(options)

    self.assertIn(
        "interval_counter",
        run_summary,
        msg=": ".join([message, ": interval_counter"]))
    # Load the generated tf file.
    dataset = tf.data.TFRecordDataset(
        run_summary["output_filename"], compression_type="GZIP")
    dataset = dataset.map(self.load_tf_example)

    if expected_total_examples == 0:
      examples = list(dataset.take(expected_total_examples))
      self.assertEmpty(examples, msg=": ".join([message, ": empty examples"]))
    else:
      # We expect at least one example here.
      examples = list(dataset.take(expected_total_examples))
      self.assertLen(
          examples,
          expected_total_examples,
          msg=": ".join([message, ": total_examples"]))
      # We don't want to do an explicit test here as it would reduce the
      # flexibility of changing values afterward. We simply want to make sure
      # values we are setting are non-empty.
      # All of the sub-functions has been checked explicitly, so we don't need
      # to check for the values again in here.
      for name, contig, active_pos, pos, ind, example, label in examples:
        self.assertIsNotNone(name, msg=": ".join([message, ": name not none"]))
        self.assertIsNotNone(
            contig, msg=": ".join([message, ": contig not none"])
        )
        self.assertIsNotNone(
            active_pos, msg=": ".join([message, ": active_pos not none"])
        )
        self.assertIsNotNone(pos, msg=": ".join([message, ": pos not none"]))
        self.assertIsNotNone(ind, msg=": ".join([message, ": ind not none"]))
        self.assertIsNotNone(
            example, msg=": ".join([message, ": example not none"])
        )
        self.assertIsNotNone(
            label, msg=": ".join([message, ": label not none"])
        )

  @parameterized.parameters(
      dict(
          active_positions=[10, 11, 12, 13, 14, 15],
          distance_threshold=5,
          positions_to_index={10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15},
          expected_filtered_list=[[10, 11, 12, 13, 14, 15]],
          message="Test 1: Simple active position filtering.",
      ),
      dict(
          active_positions=[],
          distance_threshold=5,
          positions_to_index={10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15},
          expected_filtered_list=[],
          message="Test 2: Active position filtering empty list.",
      ),
      dict(
          active_positions=[10, 11, 12, 13, 14, 15],
          distance_threshold=4,
          positions_to_index={10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15},
          expected_filtered_list=[[10, 11, 12, 13, 14], [15]],
          message="Test 3: Active position filtering with two positions.",
      ),
      dict(
          active_positions=[10, 11],
          distance_threshold=5,
          positions_to_index={10: 10, 11: 16, 12: 17, 13: 18, 14: 19, 15: 20},
          expected_filtered_list=[[10], [11]],
          message="Test 4: Active position filtering with distance.",
      ),
      dict(
          active_positions=[10],
          distance_threshold=5,
          positions_to_index={10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15},
          expected_filtered_list=[[10]],
          message="Test 5: Single active position.",
      ),
  )
  @flagsaver.flagsaver
  def test_group_nearby_active_positions(
      self,
      active_positions,
      distance_threshold,
      positions_to_index,
      expected_filtered_list,
      message,
  ):
    """Tests for group_nearby_active_positions method."""
    returned_list = region_processor.group_nearby_active_positions(
        active_positions, distance_threshold, positions_to_index
    )
    self.assertListEqual(returned_list, expected_filtered_list, msg=message)


if __name__ == "__main__":
  absltest.main()
