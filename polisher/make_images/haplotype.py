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
"""Haplotype handling for make examples."""
from typing import Dict, List
import pysam

_ALLOWED_HP_TAGS = [0, 1, 2]
_MISSING_HP_TAGS = [0]
GENERATE_IMAGES_FOR_HP_TAGS = [1, 2]


def get_read_haplotype_tag(read: pysam.AlignedSegment) -> int:
  """Return haplotype tag value of a given read.

  Args:
    read: An alignment read.

  Returns:
    A haplotype value, if read is not HP-tagged then the value is 0.

  Raises:
    ValueError: If HP-tag is not found in the allowed hp-tag list then error
    is raised.
  """
  read_haplotype_tag = read.get_tag('HP') if read.has_tag('HP') else 0
  if read_haplotype_tag not in _ALLOWED_HP_TAGS:
    raise ValueError(f'Read HP tag can be {_ALLOWED_HP_TAGS!r}.'
                     f' Found HP tag {read_haplotype_tag!r}'
                     f' in read {read.query_name!r}')
  return read_haplotype_tag


def add_non_haplotype_reads_to_bins(
    read_set: Dict[int, List[pysam.AlignedSegment]]
) -> Dict[int, List[pysam.AlignedSegment]]:
  """Add reads with missing hp tags to all tags for which we create examples.

  Missing haplotype tags are tags where either HP is 0 or none. If the value
  is none or HP tag is not set, then we consider it as HP:0.
  Meaning the phasing method didn't have confidence to put them in wither bin.
  So we assign them to both HP:1 and HP:2.

  Args:
    read_set: A dictionary where key is the haplotype tag and value is a list of
      reads with that haplotype value.

  Returns:
    A dictionary where key is the haplotype tag and value is a list of
    reads with that haplotype value with reads with missing haplotype tags to
    the bins of resolved haplotypes.
  """
  # Add reads from missing hp tags to all tags for which we generate examples.
  # For example, all HP-0 reads would appear both in HP-1 and HP-2 set.
  for missing_hp_tag in _MISSING_HP_TAGS:
    if missing_hp_tag in read_set:
      for allowed_hp_tag in GENERATE_IMAGES_FOR_HP_TAGS:
        read_set[allowed_hp_tag].extend(read_set[missing_hp_tag])
  return read_set
