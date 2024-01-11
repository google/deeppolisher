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
"""Methods for polisher inference step."""

import dataclasses
from typing import Optional, Tuple, Union
from absl import logging
import numpy as np
import pysam
import tensorflow as tf
from polisher.make_images import encoding


vocab = ''.join(encoding.get_vocab())
vocab_lookup = np.vectorize(vocab.__getitem__)


@dataclasses.dataclass
class Variant:
  contig: str
  position_start: int
  position_end: int
  ref_base: str
  alt_base: str
  quality: float
  genotype: Optional[tuple[int, int]] = None
  alt_tuple: Optional[tuple[str, str]] = None


def create_running_variant(
    contig: str,
    position: int,
    index: int,
    ref_base: str,
    pred_base: str,
    ref_base_dictionary: dict[Tuple[int, int], str],
    qual: float,
) -> Variant:
  """Returns a Variant object from given reference base and prediction base.

  Creates a variant object from prediction base and position.

  Args:
    contig: Name of contig on which variant is present.
    position: Position of the variant.
    index: Positional index for the variant. Value 0 means it's at the position,
      higher than 0 would mean an insert position.
    ref_base: Reference base observed at the position.
    pred_base: Prediction base from the sequence.
    ref_base_dictionary: Dictionary containing mapping between (position, index)
      to reference base. This is used to fetch anchor positions
    qual: Predicted quality of the variant.

  Returns:
    A Variant dataclass object explaining the variant.

  Raises:
    ValueError if (position, index) key is not present in the dictionary.
  """
  # If index is 0, that means we observed the variant against the reference
  # base (not an insertion). So, it is either a SNP or a deletion.
  if index == 0:
    # If predicted base is not GAP token then it's a simple SNP.
    if pred_base in encoding.get_valid_bases():
      return Variant(contig, position, position + 1, ref_base, pred_base, qual)
    elif pred_base == encoding.get_gap_token():
      if (position - 1, 0) not in ref_base_dictionary.keys():
        raise ValueError(
            f'Key {(position-1, 0)!r} not found in ref_base_dictionary for'
            f' contig {contig!r}, position {position!r}, index {index!r}.'
        )
      # Predicted base is a GAP, so we are inside a deletion.
      # Capture the base from the anchor.
      ref_base = ref_base_dictionary[(position - 1, 0)] + ref_base
      # For deletion the alt base is the ref base itself, we are deleting
      # the current base
      alt_base = ref_base_dictionary[(position - 1, 0)]
      return Variant(
          contig, position - 1, position + 1, ref_base, alt_base, qual
      )
    else:
      raise ValueError(f'Invalid prediction base {pred_base!r}.')
  # Index is larger than 0, so we are inside an insertion.
  else:
    if (position, 0) not in ref_base_dictionary.keys():
      raise ValueError(
          f'Key {(position, 0)!r} not found in ref_base_dictionary for'
          f' contig {contig!r}, position {position!r}, index {index!r}.'
      )
    # Capture the base from the anchor of the same position.
    ref_base = ref_base_dictionary[(position, 0)]
    # Add the insertion to the alternate base
    alt_base = ref_base_dictionary[(position, 0)] + pred_base
    return Variant(contig, position, position + 1, ref_base, alt_base, qual)


def check_valid_variant(variant: Variant) -> bool:
  """This checks if the variant we constructed contains is valid.

  A valid variant will have reference sequence different than alt variant
  and all bases will be valid bases.

  Args:
    variant: An object of Variant class.

  Returns:
    True if the variant is valid, false if not.
  """
  if variant.ref_base == variant.alt_base:
    return False

  for base in variant.ref_base:
    if base not in encoding.get_valid_bases():
      return False

  for base in variant.alt_base:
    if base not in encoding.get_valid_bases():
      return False

  return True


def get_variants_from_prediction(
    contig: str,
    reference_sequence: str,
    prediction_sequence: str,
    active_position: list[int],
    positions: list[int],
    indices: list[int],
    quality_scores: list[float],
    ref_base_dictionary: dict[Tuple[int, int], str],
) -> list[Variant]:
  """Given a reference sequence and a prediction sequence, generate variants.

  This method generates a list of variants from a given reference sequence and
  prediction sequence. The reference and predicted sequence are both padded so
  we don't have any alignment issues. For examples:
  Reference sequence:  ACGTACGT
  Prediction sequence: ACGAACGT
  Positions:           01234567
  indices:             00000000
  Would produce a variant at position 3 with ref 'T' and alt 'A'.

  Similarly for inserts:
  Reference sequence:  ACGT**ACGT
  Prediction sequence: ACGTA*ACGT
  Positions:           0123334567
  indices:             0000120000
  Would produce a variant at position 3 with ref 'T' and alt 'TA'.

  This method also takes into account where the first active position is, and
  will not generate a variant below that position.

  Args:
    contig: Name of contig on which variant is present.
    reference_sequence: A reference sequence representing assembly or ref. This
      sequence is padded for insertions.
    prediction_sequence: Predicted sequence from the model. This sequence is
      padded for insertions.
    active_position: List of positions where a candidate was observed.
    positions: List of positions for tracking genomic location.
    indices: List of indices for tracking insert positions per genomic location.
    quality_scores: Quality scores predicted from the model.
    ref_base_dictionary: Dictionary containing mapping between (position, index)
      to reference base. This is used to fetch anchor positions.

  Returns:
    A list of Variant dataclass objects explaining the difference between
    reference and predicted sequence.

  Raises:
    ValueError if it is impossible to decode the variants.
  """
  # Find out the longest sequence and we go up to that position for tracking
  # variants.
  range_limit = min(
      len(reference_sequence),
      len(prediction_sequence),
      len(positions),
      len(indices),
      len(quality_scores),
  )

  # Create a running variant to consolidate variants together, like inserts
  # or deletes that run for multiple positions.
  current_variant = None
  # Create a list to keep track of all observed variants.
  all_observed_variants = []

  # Loop over each position of the reference and prediction sequence.
  for i in range(0, range_limit):
    ref_base = reference_sequence[i].upper()
    pred_base = prediction_sequence[i].upper()
    position = positions[i]
    index = indices[i]
    qual = quality_scores[i]
    ref_base_in_anchor = ref_base_dictionary[(position, 0)]

    # Any variant upstream of active_position is not considered. This is to
    # avoid boundary issues on the left side of the window.
    # We only extend variant if it's a running variant.
    if position not in active_position and not current_variant:
      continue

    # If we are at the backbone of the reference sequence but it doesn't have
    # a valid sequence, then we do not report this. This usually happens when
    # the reference or assembly has Ns in them.
    if ref_base_in_anchor not in encoding.get_valid_bases():
      # Record any variant up to this position as they can be valid.
      if current_variant:
        if current_variant.ref_base != current_variant.alt_base:
          all_observed_variants.append(current_variant)
        current_variant = None
      continue

    # Mismatch detected, we extend a previous variant or create a new one.
    if ref_base != pred_base:
      # If current_variant is None that means we need to start a new variant.
      if not current_variant:
        current_variant = create_running_variant(
            contig,
            position,
            index,
            ref_base,
            pred_base,
            ref_base_dictionary,
            qual,
        )
      # Means we have a running variant and we also found another variant.
      else:
        # See if the running variant overlaps with the detected variant.
        # This is to handle inserts and deletes, if the current variant is at
        # the same position as the previous variant then it's an insert.
        if current_variant.position_end - position <= 1:
          # Extend an insert because the positions match and index is high.
          if index > 0 and position == current_variant.position_start:
            current_variant.alt_base = current_variant.alt_base + pred_base
            current_variant.quality = min(qual, current_variant.quality)
          # Extend a delete/GAP because it's in the next position.
          elif index == 0 and pred_base == encoding.get_gap_token():
            current_variant.ref_base = current_variant.ref_base + ref_base
            current_variant.position_end += 1
            current_variant.quality = min(qual, current_variant.quality)
          # This is a SNP and we can create a new variant.
          else:
            if check_valid_variant(current_variant):
              all_observed_variants.append(current_variant)
            current_variant = Variant(
                contig, position, position + 1, ref_base, pred_base, qual
            )
        else:
          # We should never be in this else condition as we have terminating
          # variant logic outside of this if-else statement. So raise an error.
          # Setting this value error to future proof bugs.
          raise ValueError(
              'Impossible to code variants with reference sequence'
              f' {reference_sequence!r} and prediction sequence'
              f' {prediction_sequence!r} at contig {contig !r} and position'
              f' {position!r}.'
          )
    # Check (1) if index is outside an insert, (2) the reference and prediction
    # bases are same and (3) we also have a running variant.
    # All true means we can terminate the running variant and assign none to
    # current variant.
    elif index == 0 and ref_base == pred_base and current_variant:
      if check_valid_variant(current_variant):
        all_observed_variants.append(current_variant)
      current_variant = None

  # In case there is a variant that runs over the window.
  # Example: a SNP at the very last base, or an insert that goes over to the
  # next window. This is handled during image generation where we make sure
  # that all variants are contained before the last base.
  if current_variant:
    logging.warning(
        'Skipped variant as it runs out of window %s.', current_variant
    )

  return all_observed_variants


def avg_phred(input_values: Union[np.ndarray, list[float]]) -> float:
  """Get the average phred quality given a list of phred-scaled values.

  Args:
     input_values: A numpy array containing some phred-scale values.

  Returns:
     The result of un-phredding the values, averaging them, and converting that
        average back to phred.
  """
  # Filter out base qualities that are set to -1
  # These are used to encode spacing.
  values = np.asarray(input_values)
  values = values[values >= 0]
  if not values.any():
    return 0.0
  probs = 10 ** (values / -10.0)
  avg_prob = probs.sum() / len(probs)
  avg_q = -10 * np.log10(avg_prob)
  return np.floor(avg_q)


def combine_variants_at_same_position(
    variant1: Variant, variant2: Variant
) -> Variant:
  """Combine 2 variants (from 2 haplotypes) with the same start position."""
  assert variant1.contig == variant2.contig
  assert variant1.position_start == variant2.position_start

  combined_variant = Variant(
      contig=variant1.contig,
      position_start=variant1.position_start,
      position_end=variant1.position_end,
      ref_base=variant1.ref_base,
      alt_base=variant1.alt_base,
      quality=avg_phred([variant1.quality, variant2.quality]),
  )

  if variant1.ref_base == variant2.ref_base:
    if variant1.alt_base == variant2.alt_base:
      # Same ref, same alt.
      combined_variant.genotype = (1, 1)
      return combined_variant
    else:
      # Same ref, different alts.
      combined_variant = variant1
      combined_variant.genotype = (1, 2)
      combined_variant.alt_tuple = (variant1.alt_base, variant2.alt_base)
      return combined_variant
  else:
    combined_variant.genotype = (1, 2)
    if len(variant1.ref_base) > len(variant2.ref_base):
      longer_variant = variant1
      shorter_variant = variant2
    else:
      longer_variant = variant2
      shorter_variant = variant1
    # Take the shorter variant's ALT and pad it with the longer variant's
    # extra reference sequence.
    padded_alt = (
        shorter_variant.alt_base
        + longer_variant.ref_base[len(shorter_variant.ref_base) :]
    )
    combined_variant.alt_tuple = (longer_variant.alt_base, padded_alt)
    combined_variant.ref_base = longer_variant.ref_base
    combined_variant.position_end = longer_variant.position_end
    return combined_variant


def merge_diploid_variants(
    variants_by_haplotype: list[list[Variant]],
) -> list[Variant]:
  """Merge variants together from two haplotypes.

  Args:
    variants_by_haplotype: A list of two lists containing the variants for each
      of the two haplotypes.

  Returns:
    List of merged variants such that no variants start at the same position on
        different haplotypes.
  """
  assert len(variants_by_haplotype) == 2

  variants_dicts = [{}, {}]
  for ploid, variants in enumerate(variants_by_haplotype):
    for variant in variants:
      variants_dicts[ploid][(variant.contig, variant.position_start)] = variant

  all_positions = []
  for d in variants_dicts:
    for position in d:
      all_positions.append(position)

  all_positions = sorted(list(set(all_positions)))

  merged_variants = []
  for position in all_positions:
    variant1 = variants_dicts[0].get(position, None)
    variant2 = variants_dicts[1].get(position, None)
    if variant1 and variant2:
      combined_variant = combine_variants_at_same_position(variant1, variant2)
      merged_variants.append(combined_variant)
    elif variant1:
      merged_variants.append(variant1)
    elif variant2:
      merged_variants.append(variant2)
    else:
      raise ValueError(
          'merge_diploid_variants seems to be running with no '
          'variants. This should have been prevented upstream.'
      )
  return merged_variants


def variants_from_example(
    example_i: int,
    batch: dict[str, tf.Tensor],
    batch_predictions: np.ndarray,
    batch_quality_scores: np.ndarray,
    fasta_file: pysam.FastaFile,
) -> list[Variant]:
  """Postprocess one window (example) to catalogue any variants found within.

  Args:
    example_i: Index of which example to process within the batch.
    batch: A batch of examples, keyed by feature names and within each of those
      it is an iterable of examples.
    batch_predictions: Predictions for the batch of examples.
    batch_quality_scores: Quality scores for the batch of examples.
    fasta_file: A pysam handle to a FASTA file.

  Returns:
    A list of variant(s) found in the given example.
  """
  contig = batch['contig'][example_i]
  active_position = batch['active_position'][example_i]
  encoded_reference = batch['encoded_reference'][example_i]
  reference_positions = batch['reference_positions'][example_i]
  reference_indices = batch['reference_indices'][example_i]
  predicted_vector = batch_predictions[example_i]
  quality_scores = batch_quality_scores[example_i]

  contig_name = contig.numpy()[0].decode('utf-8')
  reference_sequence = ''.join(
      vocab_lookup(np.squeeze(encoded_reference, axis=0))
  )
  ref_pos_start = max(0, min(reference_positions.numpy().tolist()) - 5)
  ref_pos_end = max(reference_positions.numpy().tolist()) + 5
  ref_sequence = fasta_file.fetch(contig_name, ref_pos_start, ref_pos_end)
  ref_base_dictionary = {}
  for i in range(0, len(ref_sequence)):
    ref_base_dictionary[(ref_pos_start + i, 0)] = ref_sequence[i]

  # Automatically determine ploidy from output shape:
  if predicted_vector.shape == (2, 100):
    ploidy = 2  # diploid.
  else:
    ploidy = 1  # haploid.

  if ploidy == 1:
    prediction_sequence = ''.join(
        np.vectorize(vocab.__getitem__)(predicted_vector)
    )
    variants = get_variants_from_prediction(
        contig=contig_name,
        reference_sequence=reference_sequence,
        prediction_sequence=prediction_sequence,
        active_position=active_position.numpy(),
        positions=reference_positions.numpy(),
        indices=reference_indices.numpy(),
        quality_scores=quality_scores,
        ref_base_dictionary=ref_base_dictionary,
    )
    return variants
  elif ploidy == 2:
    # DeepLoid has 2 output sequences at each position:
    variants_by_haplotype = [[], []]
    for ploid in range(2):
      prediction_sequence = ''.join(vocab_lookup(predicted_vector[ploid]))
      variants_by_haplotype[ploid] = get_variants_from_prediction(
          contig=contig_name,
          reference_sequence=reference_sequence,
          prediction_sequence=prediction_sequence,
          active_position=active_position,
          positions=reference_positions.numpy(),
          indices=reference_indices.numpy(),
          quality_scores=quality_scores[ploid],
          ref_base_dictionary=ref_base_dictionary,
      )
    return merge_diploid_variants(variants_by_haplotype)
  else:
    raise ValueError('Ploidy must be 1 or 2.')
