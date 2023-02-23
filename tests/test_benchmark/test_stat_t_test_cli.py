# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for stat_t_test_cli.py"""

import scipy

import pytest
from pytest import approx

from tensorflow_io.python.experimental.benchmark.stat_t_test_cli import \
  run_welchs_ttest, MEAN, STDDEV, ROUNDS


@pytest.mark.parametrize(
  ["mean1", "std1", "n1", "mean2", "std2", "n2", "is_faster"], [
    (10., 1., 30, 10., 1., 30, True),
    (10., 1., 30, 10., 1., 30, False),
    (6.0, 0.5, 40, 5.9, 0.4, 30, True),
    (6.0, 0.5, 40, 5.9, 0.4, 30, False),
    (9.75, 1.0, 100, 10.0, 1.0, 100, True),
    (9.75, 1.0, 100, 10.0, 1.0, 100, False),
  ]
)
def test_p_value(mean1, std1, n1, mean2, std2, n2, is_faster):
  """
  This test case is to make sure the p value and t stat computed in
  run_welchs_ttest is correct by comparing with the values generated
  from scipy.stats.ttest_ind_from_stats.
  """
  alternative = 'less' if is_faster else 'greater'
  expected_t_stat, expected_p_value = scipy.stats.ttest_ind_from_stats(
    mean1=mean1, std1=std1, nobs1=n1,
    mean2=mean2, std2=std2, nobs2=n2,
    equal_var=False,  # False for Welch's t-test
    alternative=alternative
  )

  stat1 = {
    MEAN: mean1,
    STDDEV: std1,
    ROUNDS: n1
  }
  stat2 = {
    MEAN: mean2,
    STDDEV: std2,
    ROUNDS: n2
  }

  alpha = 0.05
  ttest_result = run_welchs_ttest(stat1, stat2, alpha, is_faster)

  assert expected_p_value == approx(ttest_result.p_value)
  assert expected_t_stat == approx(ttest_result.t_stat)
