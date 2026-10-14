import torch

from acoustic_baseline.model import AcousticCNN1D


def test_output_shape():
    model = AcousticCNN1D()
    x = torch.zeros(8, 1, 1024)
    y = model(x)
    assert y.shape == (8, 6)


def test_feature_shape_and_dimension():
    model = AcousticCNN1D()
    assert model.feature_shape == (64, 55)
    assert model.feature_dim == 3520


def test_feature_extractor_shape():
    model = AcousticCNN1D()
    x = torch.zeros(4, 1, 1024)
    z = model.extract_features(x)
    assert z.shape == (4, 3520)
