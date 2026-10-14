from acoustic_baseline.training import paper_lr


def test_learning_rate_starts_at_base_rate():
    assert abs(paper_lr(0, 50) - 0.0005) < 1e-12


def test_learning_rate_decreases():
    assert paper_lr(25, 50) < paper_lr(0, 50)
    assert paper_lr(49, 50) < paper_lr(25, 50)
