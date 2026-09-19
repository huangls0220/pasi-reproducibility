import math
import numpy as np
import pandas as pd
import pytest
from scripts.run_e30_episode_risk import bounded_kl_upper, kl, selection_from_bounds
from scripts.run_e23_guardrail_frontier import PROFILES, profile_weight
from scripts.run_e20_envelope_tradeoff import SCENARIOS


def test_zero_and_all_one():
    assert bounded_kl_upper([0]*15, .05/30) == pytest.approx(1-(.05/30)**(1/15))
    assert bounded_kl_upper([1]*15, .01) == 1


def test_fractional_episode_losses_inversion():
    x = [0, .1, .2, 0, .5]
    u = bounded_kl_upper(x, .01)
    assert len(x)*kl(np.mean(x),u) == pytest.approx(math.log(100))
    assert u > np.mean(x)


def test_no_assignment_pseudoreplication():
    # Two episodes remain two samples regardless of their service counts.
    a = [100,10000]
    v = [1,100]
    losses = np.array(v)/a
    assert bounded_kl_upper(losses,.05) > bounded_kl_upper([.01]*100,.05)


def test_monotonicity():
    assert bounded_kl_upper([.1]*10,.01) > bounded_kl_upper([.1]*100,.01)
    assert bounded_kl_upper([.1]*10,.01) > bounded_kl_upper([.1]*10,.05)


@pytest.mark.parametrize('losses,tail', [([], .05), ([float('nan')],.05), ([-.1],.05), ([1.1],.05), ([0],0)])
def test_invalid_inputs(losses, tail):
    with pytest.raises(ValueError):
        bounded_kl_upper(losses,tail)


def fixture_bounds(value):
    return pd.DataFrame([{'scenario':s,'profile':p,'global_guardrail_weight':profile_weight(p),
                          'episode_risk_upper':value}
                         for s in SCENARIOS for p in PROFILES])


def test_no_certification_returns_structural_not_statistical():
    frame = fixture_bounds(bounded_kl_upper([0]*15,.05/30))
    got = selection_from_bounds(frame)
    assert got['selected_g']==1 and not got['statistical_pass']


def test_selects_smallest_valid_without_assuming_monotonicity():
    frame = fixture_bounds(.02)
    frame.loc[frame.global_guardrail_weight==.25,'episode_risk_upper']=.004
    assert selection_from_bounds(frame)['selected_g']==.25
    assert selection_from_bounds(frame)['statistical_pass']


def test_missing_cell_rejected():
    with pytest.raises(ValueError):
        selection_from_bounds(fixture_bounds(.01).iloc[1:])
