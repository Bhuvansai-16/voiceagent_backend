"""Filler wording. Plan section 148 wants variation so it does not sound robotic."""

import random

from backend.worker import filler


def test_first_phrase_is_short_enough_to_fit_the_gap():
    # These play over a sub-second gap. Anything long is worse than the silence.
    assert all(len(p) <= 30 for p in filler.PHRASES), filler.PHRASES


def test_delay_fits_under_the_plans_800ms_acceptance_case():
    """Plan section 154 wants filler, not dead air, when the backend takes 800ms.
    A delay at or above that would never fire in the case it exists for."""
    assert filler.DELAY < 0.8


def test_step_zero_picks_from_the_opening_pool():
    assert filler.phrases(random.Random(0))(0) in filler.PHRASES


def test_later_steps_use_follow_ups_not_the_opener_again():
    pick = filler.phrases(random.Random(0))
    assert [pick(step) for step in (1, 2, 3)] == list(filler.FOLLOW_UPS)


def test_follow_ups_wrap_instead_of_running_off_the_end():
    pick = filler.phrases(random.Random(0))
    assert pick(1 + len(filler.FOLLOW_UPS)) == filler.FOLLOW_UPS[0]


def test_different_calls_do_not_always_open_with_the_same_line():
    openers = {filler.phrases(random.Random(seed))(0) for seed in range(20)}
    assert len(openers) > 1, "filler always opens the same way, which is the robotic sound"
