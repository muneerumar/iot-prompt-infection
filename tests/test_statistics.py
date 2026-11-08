import random

import numpy as np
import pytest
from scipy.stats import binomtest, bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from iotpi.stats import cluster_bootstrap, holm, mcnemar_exact


def test_exact_mcnemar_matches_scipy_and_statsmodels():
    rng = random.Random(23)
    for _ in range(80):
        a = [rng.choice([False, True]) for _ in range(rng.randrange(1, 250))]
        b = [rng.choice([False, True]) for _ in a]
        result = mcnemar_exact(a, b)
        discordant = result['n01'] + result['n10']
        reference = binomtest(result['n01'], discordant, .5).pvalue if discordant else 1.0
        table = np.array([[sum(not x and not y for x,y in zip(a,b)), result['n01']],
                          [result['n10'], sum(x and y for x,y in zip(a,b))]])
        assert result['p'] == pytest.approx(reference)
        assert result['p'] == pytest.approx(mcnemar(table, exact=True).pvalue)


def test_holm_matches_statsmodels_on_random_families():
    rng = np.random.default_rng(32)
    for size in range(1, 30):
        values = rng.uniform(0, .15, size)
        result = holm(dict(enumerate(values)))
        adjusted = multipletests(values, method='holm')[1]
        assert np.allclose([result[i] for i in range(size)], adjusted)


def test_cluster_bootstrap_agrees_with_scipy_cluster_resampling():
    rng = np.random.default_rng(34)
    for trial in range(8):
        groups = [rng.binomial(1, rng.uniform(.1,.8), rng.integers(2,12)) for _ in range(14)]
        sums, counts = np.array([x.sum() for x in groups]), np.array([len(x) for x in groups])
        values = [float(v) for group in groups for v in group]
        clusters = [str(i) for i,group in enumerate(groups) for _ in group]
        estimate, lo, hi = cluster_bootstrap(values, clusters, reps=6000, seed=trial)
        def ratio(total, count, axis=-1):
            return total.sum(axis=axis) / count.sum(axis=axis)
        reference = bootstrap((sums,counts), ratio, paired=True, method='percentile',
                              n_resamples=6000, rng=np.random.default_rng(trial))
        assert estimate == pytest.approx(sums.sum()/counts.sum())
        # Independent RNGs give independent Monte Carlo draws of the same distribution.
        assert abs(lo-reference.confidence_interval.low) < .025
        assert abs(hi-reference.confidence_interval.high) < .025


def test_paired_statistics_reject_unmatched_inputs():
    with pytest.raises(ValueError): mcnemar_exact([True,False], [True])
    with pytest.raises(ValueError): cluster_bootstrap([1,0], ['a'])


def test_predictive_interval_retains_binary_episode_variation():
    from iotpi.predict import predictive_cohort_means
    rows=[{"predictive_outcomes":[[False,True],[True,False]],"pred_draws":[.5,.5]}]
    cohort=predictive_cohort_means(rows)
    assert np.percentile(cohort,[2.5,97.5]).tolist()==[0,1]
    assert np.percentile(rows[0]['pred_draws'],[2.5,97.5]).tolist()==[.5,.5]
    with pytest.raises(ValueError): predictive_cohort_means([{"predictive_outcomes":[[.5,.5]]}])
