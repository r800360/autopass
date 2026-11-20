"""Numerical geometry checks with simulator-free fixture data."""
import pytest

np = pytest.importorskip("numpy")

from autopass.carla_backend.frenet import Frenet
from autopass.carla_backend.perception import decode_depth, decode_instances, decode_rgb


def test_ego_projection_uses_local_lane_even_if_the_full_route_bends_back():
    xy = np.array([[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [4, 2], [2, 2], [0, 2]])
    curve = Frenet(xy, np.array([0, 0, 0, 0, 90, 180, 180, 180]), 0, False, 2)
    s, d = curve.to_sd([[2.5, 0.25]], near_s=2.0)
    assert float(s[0]) == pytest.approx(2.5)
    assert float(d[0]) == pytest.approx(.25)
    assert curve.to_xy(float(s[0]), float(d[0])) == pytest.approx((2.5, .25))


def test_carla_colors_ids_and_metric_depth():
    class Image:
        width, height = 2, 1
        raw_data = bytes([1, 5, 14, 255, 0, 15, 14, 255])

    tags, ids = decode_instances(Image())
    assert tags.tolist() == [[14, 14]] and ids.tolist() == [[261, 15]]
    assert decode_rgb(Image()).tolist() == [[[14, 5, 1], [14, 15, 0]]]
    assert float(decode_depth(Image())[0, 0]) == pytest.approx(1000 * (65536 + 5*256 + 14) / (256**3-1))
