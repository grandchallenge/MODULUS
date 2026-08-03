from modulus.online import AnytimeHoeffdingCS, RegretTracker


def test_regret_tracker_exact_values() -> None:
    tracker = RegretTracker(num_comparators=2)
    tracker.update(0.4, [0.2, 0.5])
    tracker.update(0.4, [0.7, 0.1])
    snapshot = tracker.snapshot()
    assert snapshot.cumulative_loss == 0.8
    assert snapshot.best_fixed_comparator_loss == 0.6
    assert abs(snapshot.static_regret - 0.2) < 1e-12
    assert abs(snapshot.dynamic_comparator_loss - 0.3) < 1e-12
    assert abs(snapshot.dynamic_regret - 0.5) < 1e-12
    assert abs(tracker.best_k_switch_comparator_loss(1) - 0.3) < 1e-12
    assert abs(tracker.tracking_regret(1) - 0.5) < 1e-12


def test_maximum_interval_regret() -> None:
    tracker = RegretTracker(num_comparators=2)
    tracker.update(1.0, [0.0, 1.0])
    tracker.update(0.0, [1.0, 0.0])
    assert tracker.maximum_interval_static_regret() == 1.0


def test_anytime_hoeffding_sequence_contracts() -> None:
    cs = AnytimeHoeffdingCS(0.0, 1.0, delta=0.05)
    first = cs.update(0.5)
    for _ in range(999):
        last = cs.update(0.5)
    assert first == (0.0, 1.0)
    assert last[0] < 0.5 < last[1]
    assert (last[1] - last[0]) < 0.2
