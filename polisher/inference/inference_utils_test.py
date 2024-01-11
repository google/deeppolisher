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
"""Tests for inference_utils."""

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from polisher.inference import inference_utils

Variant = inference_utils.Variant


class InferenceUtilsTest(parameterized.TestCase):
  """Tests for inference utils."""

  @parameterized.parameters(
      dict(
          contig="chr1",
          position=0,
          index=0,
          ref_base="A",
          pred_base="C",
          ref_base_dictionary={(0, 0): "A", (1, 0): "C", (2, 0): "G"},
          qual=10,
          expected_variant=inference_utils.Variant("chr1", 0, 1, "A", "C", 10),
          message="Simple SNP test.",
      ),
      dict(
          contig="chr1",
          position=0,
          index=1,
          ref_base="*",
          pred_base="C",
          ref_base_dictionary={(0, 0): "A", (1, 0): "C", (2, 0): "G"},
          qual=10,
          expected_variant=inference_utils.Variant("chr1", 0, 1, "A", "AC", 10),
          message="Simple INS test.",
      ),
      dict(
          contig="chr1",
          position=1,
          index=0,
          ref_base="C",
          pred_base="*",
          ref_base_dictionary={(0, 0): "A", (1, 0): "C", (2, 0): "G"},
          qual=10,
          expected_variant=inference_utils.Variant("chr1", 0, 2, "AC", "A", 10),
          message="Simple DEL test.",
      ),
  )
  @flagsaver.flagsaver
  def test_create_running_variant(
      self,
      contig,
      position,
      index,
      ref_base,
      pred_base,
      ref_base_dictionary,
      qual,
      expected_variant,
      message,
  ):
    """Test create_running_variant method."""
    generated_variant = inference_utils.create_running_variant(
        contig, position, index, ref_base, pred_base, ref_base_dictionary, qual
    )
    self.assertEqual(generated_variant, expected_variant, msg=message)

  @parameterized.parameters(
      dict(
          contig="chr1",
          position=4,
          index=0,
          ref_base="C",
          pred_base="*",
          ref_base_dictionary={(0, 0): "A", (1, 0): "C", (2, 0): "G"},
          qual=10,
          exception_msg=(
              "Key (3, 0) not found in ref_base_dictionary for contig 'chr1',"
              " position 4, index 0."
          ),
          message="Exception test for deletion anchor.",
      ),
      dict(
          contig="chr1",
          position=3,
          index=1,
          ref_base="*",
          pred_base="C",
          ref_base_dictionary={(0, 0): "A", (1, 0): "C", (2, 0): "G"},
          qual=10,
          exception_msg=(
              "Key (3, 0) not found in ref_base_dictionary for contig 'chr1',"
              " position 3, index 1."
          ),
          message="Exception test for insertion anchor.",
      ),
      dict(
          contig="chr1",
          position=3,
          index=0,
          ref_base="A",
          pred_base="K",
          ref_base_dictionary={(0, 0): "A", (1, 0): "C", (2, 0): "G"},
          qual=10,
          exception_msg="Invalid prediction base 'K'.",
          message="Exception test for insertion anchor.",
      ),
  )
  @flagsaver.flagsaver
  def test_create_running_variant_exceptions(
      self,
      contig,
      position,
      index,
      ref_base,
      pred_base,
      ref_base_dictionary,
      qual,
      exception_msg,
      message,
  ):
    """Test the exception captures of create_running_variant method."""
    with self.assertRaisesWithLiteralMatch(
        ValueError, exception_msg, msg=message
    ):
      inference_utils.create_running_variant(
          contig,
          position,
          index,
          ref_base,
          pred_base,
          ref_base_dictionary,
          qual,
      )

  @parameterized.parameters(
      dict(
          contig="chr1",
          reference_sequence="AAACC",
          prediction_sequence="AATCC",
          active_position=[3],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "A", "T", 5)
          ],
          message="Single SNP test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="AAACC",
          prediction_sequence="AATTC",
          active_position=[3, 4],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "A", "T", 5),
              inference_utils.Variant("chr1", 4, 5, "C", "T", 5),
          ],
          message="Multiple SNP test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="AAACCG",
          prediction_sequence="TATCTG",
          active_position=[3, 4, 5],
          positions=[1, 2, 3, 4, 5, 6],
          indices=[0, 0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 6, 6],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "A", "T", 5),
              inference_utils.Variant("chr1", 5, 6, "C", "T", 6),
          ],
          message="Multiple SNP test at different positions.",
      ),
      dict(
          # In this test, there's a variant at the last position that will be
          # discarded.
          contig="chr1",
          reference_sequence="AAACC",
          prediction_sequence="AATTT",
          active_position=[3, 4, 5],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "A", "T", 5),
              inference_utils.Variant("chr1", 4, 5, "C", "T", 5),
          ],
          message="Multiple SNP test with one overflow.",
      ),
      dict(
          contig="chr1",
          reference_sequence="ACG**TT",
          prediction_sequence="ACGT*TT",
          active_position=[3],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "G", "GT", 4)
          ],
          message="Single base insert test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="ACG**TT",
          prediction_sequence="ACGTTTT",
          active_position=[3],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "G", "GTT", 4)
          ],
          message="Two base insert test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="ACG**TT",
          prediction_sequence="ACGTTCT",
          active_position=[3],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "G", "GTT", 4),
              inference_utils.Variant("chr1", 4, 5, "T", "C", 6),
          ],
          message="Two base insert test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="ACG**TT",
          prediction_sequence="ACG*TTT",
          active_position=[3],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 4, "G", "GT", 5)
          ],
          message="One base padded insert test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="ACG**",
          prediction_sequence="ACGTT",
          active_position=[3],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[],
          message="Insert out of window test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="AAACC",
          prediction_sequence="AA*CC",
          active_position=[3],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 2, 4, "AA", "A", 5)
          ],
          message="Single base deletion test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="AAACC",
          prediction_sequence="AA**C",
          active_position=[2, 3],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 2, 5, "AAC", "A", 5)
          ],
          message="Multiple base deletion test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="AAACC",
          prediction_sequence="A*A*C",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 1, 3, "AA", "A", 5),
              inference_utils.Variant("chr1", 3, 5, "AC", "A", 5),
          ],
          message="Multiple deletions within window test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="AAACC",
          prediction_sequence="AA***",
          active_position=[3],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[],
          message="Deletion overflow test.",
      ),
      #####################################
      # Tests for complex variants (MNPs) #
      #####################################
      dict(
          # This is not an MNP, we still report each SNP independently.
          contig="chr1",
          reference_sequence="ACGTA",
          prediction_sequence="ATTAA",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 2, 3, "C", "T", 5),
              inference_utils.Variant("chr1", 3, 4, "G", "T", 5),
              inference_utils.Variant("chr1", 4, 5, "T", "A", 5),
          ],
          message="MNP SNP test.",
      ),
      dict(
          # This generally means consolidating INDELs.
          contig="chr1",
          reference_sequence="ACG**CT",
          prediction_sequence="ACG*T*T",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 5, "GC", "GT", 5)
          ],
          message="INDEL with same anchor test.",
      ),
      dict(
          # Here we end up with the same allele as the reference, so we don't
          # report it. The ref would be GC and alt would also be GC.
          contig="chr1",
          reference_sequence="ACG**CT",
          prediction_sequence="ACG*C*T",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[],
          message="INDEL with same anchor test ends up having same allele.",
      ),
      dict(
          contig="chr1",
          reference_sequence="ACG**CT",
          prediction_sequence="ATA*T*T",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7],
          expected_variants=[
              inference_utils.Variant("chr1", 2, 3, "C", "T", 2),
              inference_utils.Variant("chr1", 3, 5, "GC", "AT", 3),
          ],
          message="INDEL with same anchor test.",
      ),
      dict(
          contig="chr1",
          reference_sequence="ACG**CTAG",
          prediction_sequence="ACGTT*TTG",
          active_position=[2, 3, 4, 5, 6, 7],
          positions=[1, 2, 3, 3, 3, 4, 5, 6, 7],
          indices=[0, 0, 0, 1, 2, 0, 0, 0, 0],
          quality_scores=[1, 2, 3, 4, 5, 6, 7, 8, 9],
          expected_variants=[
              inference_utils.Variant("chr1", 3, 5, "GC", "GTT", 4),
              inference_utils.Variant("chr1", 6, 7, "A", "T", 8),
          ],
          message="MNP with deletion right after insertion.",
      ),
      #####################################
      # Invalid reference base test       #
      #####################################
      dict(
          # This is not an MNP, we still report each SNP independently.
          contig="chr1",
          reference_sequence="ACNTA",
          prediction_sequence="ATTAA",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 2, 3, "C", "T", 5),
              inference_utils.Variant("chr1", 4, 5, "T", "A", 5),
          ],
          message="Invalid reference base test with SNPs.",
      ),
      dict(
          # This is not an MNP, we still report each SNP independently.
          contig="chr1",
          reference_sequence="ACN**TA",
          prediction_sequence="ATTTTAA",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 3, 3, 4, 5],
          indices=[0, 0, 0, 1, 2, 0, 0],
          quality_scores=[5, 5, 5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 2, 3, "C", "T", 5),
              inference_utils.Variant("chr1", 4, 5, "T", "A", 5),
          ],
          message="Invalid reference base test with INDEL anchor.",
      ),
      dict(
          # This is not an MNP, we still report each SNP independently.
          contig="chr1",
          reference_sequence="ACNTA",
          prediction_sequence="AT**A",
          active_position=[2, 3, 4, 5],
          positions=[1, 2, 3, 4, 5],
          indices=[0, 0, 0, 0, 0],
          quality_scores=[5, 5, 5, 5, 5],
          expected_variants=[
              inference_utils.Variant("chr1", 2, 3, "C", "T", 5),
          ],
          message="Invalid reference base test with deletion multiple bases.",
      ),
  )
  @flagsaver.flagsaver
  def test_get_variants_from_prediction(
      self,
      contig,
      reference_sequence,
      prediction_sequence,
      active_position,
      positions,
      indices,
      quality_scores,
      expected_variants,
      message,
  ):
    """Test get_variant_from_prediction method."""
    # For simplicity of writing the tests, we are building this dictionary here.
    ref_base_dictionary = {}
    for pos, index, base in zip(positions, indices, reference_sequence):
      ref_base_dictionary[(pos, index)] = base
    predicted_variants = inference_utils.get_variants_from_prediction(
        contig,
        reference_sequence,
        prediction_sequence,
        active_position,
        positions,
        indices,
        quality_scores,
        ref_base_dictionary,
    )
    self.assertListEqual(predicted_variants, expected_variants, msg=message)

  @parameterized.parameters(
      dict(
          ref1="ACG",
          alt1="A",
          ref2="A",
          alt2="G",
          final_ref="ACG",
          final_alts=("A", "GCG"),
      ),
      dict(
          ref1="T",
          alt1="TGAAGATGA",
          ref2="T",
          alt2="TGG",
          final_ref="T",
          final_alts=("TGAAGATGA", "TGG"),
      ),
      dict(
          ref1="TTT",
          alt1="G",
          ref2="T",
          alt2="C",
          final_ref="TTT",
          final_alts=("G", "CTT"),
      ),
      dict(
          ref1="G",
          alt1="C",
          ref2="G",
          alt2="T",
          final_ref="G",
          final_alts=("C", "T"),
      ),
      dict(
          ref1="G",
          alt1="C",
          ref2="GG",
          alt2="TT",
          final_ref="GG",
          final_alts=("CG", "TT"),
      ),
      dict(
          ref1="TCT",
          alt1="T",
          ref2="T",
          alt2="TCTCTCTC",
          final_ref="TCT",
          final_alts=("T", "TCTCTCTCCT"),
      ),
      dict(
          ref1="C",
          alt1="CGTGTGTGT",
          ref2="CGTGT",
          alt2="C",
          final_ref="CGTGT",
          final_alts=("CGTGTGTGTGTGT", "C"),
      ),
  )
  def test_combine_variants_at_same_position(
      self, ref1, alt1, ref2, alt2, final_ref, final_alts
  ):
    variant1 = inference_utils.Variant(
        contig="chr20",
        position_start=1000,
        position_end=10001,
        ref_base=ref1,
        alt_base=alt1,
        quality=30,
    )
    variant2 = inference_utils.Variant(
        contig="chr20",
        position_start=1000,
        position_end=10001,
        ref_base=ref2,
        alt_base=alt2,
        quality=30,
    )

    output = inference_utils.combine_variants_at_same_position(
        variant1, variant2
    )
    self.assertEqual(output.ref_base, final_ref)
    self.assertCountEqual(output.alt_tuple, final_alts)

  @parameterized.parameters(
      dict(
          inputs=[
              [
                  Variant(
                      contig="chr20",
                      position_start=88107,
                      position_end=88108,
                      ref_base="T",
                      alt_base="C",
                      quality=27,
                      genotype=None,
                      alt_tuple=None,
                  )
              ],
              [],
          ],
          expected_output=[
              Variant(
                  contig="chr20",
                  position_start=88107,
                  position_end=88108,
                  ref_base="T",
                  alt_base="C",
                  quality=27,
                  genotype=None,
                  alt_tuple=None,
              )
          ],
      ),
      dict(
          inputs=[
              [
                  Variant(
                      contig="chr20",
                      position_start=106774,
                      position_end=106775,
                      ref_base="A",
                      alt_base="C",
                      quality=34,
                      genotype=(1, 1),
                      alt_tuple=None,
                  )
              ],
              [
                  Variant(
                      contig="chr20",
                      position_start=106774,
                      position_end=106775,
                      ref_base="A",
                      alt_base="C",
                      quality=32,
                      genotype=None,
                      alt_tuple=None,
                  )
              ],
          ],
          expected_output=[
              Variant(
                  contig="chr20",
                  position_start=106774,
                  position_end=106775,
                  ref_base="A",
                  alt_base="C",
                  quality=32.0,
                  genotype=(1, 1),
                  alt_tuple=None,
              )
          ],
      ),
      dict(
          inputs=[
              [
                  Variant(
                      contig="chr20",
                      position_start=157931,
                      position_end=157937,
                      ref_base="TGTGTG",
                      alt_base="T",
                      quality=43,
                      genotype=(1, 2),
                      alt_tuple=("T", "TTG"),
                  ),
                  Variant(
                      contig="chr20",
                      position_start=157978,
                      position_end=157979,
                      ref_base="A",
                      alt_base="C",
                      quality=4,
                      genotype=None,
                      alt_tuple=None,
                  ),
              ],
              [
                  Variant(
                      contig="chr20",
                      position_start=157931,
                      position_end=157935,
                      ref_base="TGTG",
                      alt_base="T",
                      quality=22,
                      genotype=None,
                      alt_tuple=None,
                  ),
                  Variant(
                      contig="chr20",
                      position_start=157959,
                      position_end=157960,
                      ref_base="T",
                      alt_base="C",
                      quality=8,
                      genotype=None,
                      alt_tuple=None,
                  ),
                  Variant(
                      contig="chr20",
                      position_start=157961,
                      position_end=157962,
                      ref_base="T",
                      alt_base="C",
                      quality=8,
                      genotype=None,
                      alt_tuple=None,
                  ),
              ],
          ],
          expected_output=[
              Variant(
                  contig="chr20",
                  position_start=157931,
                  position_end=157937,
                  ref_base="TGTGTG",
                  alt_base="T",
                  quality=24.0,
                  genotype=(1, 2),
                  alt_tuple=("T", "TTG"),
              ),
              Variant(
                  contig="chr20",
                  position_start=157959,
                  position_end=157960,
                  ref_base="T",
                  alt_base="C",
                  quality=8,
                  genotype=None,
                  alt_tuple=None,
              ),
              Variant(
                  contig="chr20",
                  position_start=157961,
                  position_end=157962,
                  ref_base="T",
                  alt_base="C",
                  quality=8,
                  genotype=None,
                  alt_tuple=None,
              ),
              Variant(
                  contig="chr20",
                  position_start=157978,
                  position_end=157979,
                  ref_base="A",
                  alt_base="C",
                  quality=4,
                  genotype=None,
                  alt_tuple=None,
              ),
          ],
      ),
  )
  def test_merge_diploid_variants(self, inputs, expected_output):
    observed_output = inference_utils.merge_diploid_variants(inputs)
    # "Count" here just means the order doesn't matter.
    self.assertCountEqual(observed_output, expected_output)


if __name__ == "__main__":
  absltest.main()
