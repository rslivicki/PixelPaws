"""Exclusion rules (licking overrides grooming): the overridden behavior must
not be called on the overriding behavior's frames, even when gap bridging
would join the bouts on either side of it."""
import numpy as np

from prediction_pipeline import apply_smoothing, smooth_with_exclusions

CLF = {'best_thresh': 0.5, 'min_bout': 10, 'min_after_bout': 1, 'max_gap': 40}


def _grooming_around_licking():
    p = np.full(300, 0.05)
    p[50:150] = 0.95           # grooming ...
    p[170:250] = 0.95          # ... and again after a short gap
    lick = np.zeros(300, dtype=bool)
    lick[150:170] = True       # the licking bout in the gap
    return p, lick


def test_bridging_does_not_refill_overriding_frames():
    p, lick = _grooming_around_licking()
    # zeroing the probabilities alone: max_gap bridges the 20-frame gap
    zeroed = p.copy()
    zeroed[lick] = 0.0
    assert apply_smoothing(zeroed, CLF)[lick].all()
    # with the helper, no grooming call on a licking frame
    yp, pred = smooth_with_exclusions(p, CLF, 'bout_filters', lick)
    assert not pred[lick].any()
    assert (yp[lick] == 0).all()
    # the grooming on either side is kept
    assert pred[50:150].all() and pred[170:250].all()


def test_no_mask_is_plain_smoothing():
    p, _ = _grooming_around_licking()
    for ex in (None, np.zeros(300, dtype=bool)):
        yp, pred = smooth_with_exclusions(p, CLF, 'bout_filters', ex)
        assert np.array_equal(pred, apply_smoothing(p, CLF))
        assert np.array_equal(yp, p)


def test_mask_shorter_or_longer_than_probabilities():
    p, lick = _grooming_around_licking()
    _, pred = smooth_with_exclusions(p, CLF, 'bout_filters', lick[:160])
    assert not pred[150:160].any()
    _, pred = smooth_with_exclusions(p, CLF, 'bout_filters', np.r_[lick, np.ones(50, bool)])
    assert len(pred) == 300 and not pred[lick].any()


# ---------------------------------------------------------------------------
# Priority order: the batch resolves overlaps in the manifest's priority_order;
# the Multi-Classifier and Sequencing tabs rank states with their own template.
# They must agree, or totals and state occupancy would tell different stories.

import json
import os
import types

_MAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "pixelpaws_global_classifier_encyclopedia", "manifest.json")


def _prio():
    with open(_MAN, encoding="utf-8") as fh:
        return json.load(fh)["priority_order"]


def test_manifest_priority_matches_tab_templates():
    import multi_classifier_tab as MC
    import sequencing_tab as SQ
    prio = _prio()
    names = {c["name"] for c in json.load(open(_MAN, encoding="utf-8"))["classifiers"]}
    assert set(prio) == names          # every shipped classifier has a rank
    for mod in (MC, SQ):
        ranks = [mod._template_rank(b) for b in prio]
        assert ranks == sorted(ranks), (mod.__name__, list(zip(prio, ranks)))


def test_rules_follow_priority_order():
    import PixelPaws_GUI as G
    stub = types.SimpleNamespace()
    stub.bundled_priority_order = lambda: G.PixelPawsGUI.bundled_priority_order(stub)
    rules = G.PixelPawsGUI.bundled_suppression_rules(stub)
    prio = _prio()
    assert prio[0] not in rules                       # licking is never overridden
    for i, b in enumerate(prio[1:], start=1):
        assert set(prio[:i]) <= set(rules[b]), b      # everything above overrides it
        assert not set(prio[i:]) & set(rules[b]), b   # nothing below does
