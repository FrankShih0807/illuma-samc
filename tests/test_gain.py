"""Tests for gain sequences."""

import math

import torch

from illuma_samc.gain import GainSequence


class TestOneOverT:
    def test_warmup_phase(self):
        g = GainSequence("1/t", t0=100)
        # During warmup (t < t0), gain = t0/t0 = 1
        assert g(1) == 1.0
        assert g(50) == 1.0
        assert g(100) == 1.0

    def test_decay_phase(self):
        g = GainSequence("1/t", t0=100)
        assert g(200) == 100.0 / 200
        assert g(1000) == 100.0 / 1000

    def test_monotone_decreasing_after_warmup(self):
        g = GainSequence("1/t", t0=10)
        vals = [g(t) for t in range(10, 200)]
        for a, b in zip(vals, vals[1:]):
            assert a >= b


class TestLog:
    def test_warmup_phase(self):
        g = GainSequence("log", t0=100)
        # t * log(t+e) is small early on, so gain = t0/t0 = 1
        assert g(1) == 1.0

    def test_decay_phase(self):
        g = GainSequence("log", t0=10)
        val = g(1000)
        expected = 10.0 / (1000 * math.log(1000 + math.e))
        assert abs(val - expected) < 1e-12

    def test_decays_faster_than_one_over_t(self):
        g_inv = GainSequence("1/t", t0=100)
        g_log = GainSequence("log", t0=100)
        # At large t, log schedule decays faster
        assert g_log(10000) < g_inv(10000)


class TestRamp:
    def test_warmup_constant(self):
        g = GainSequence("ramp", rho=1.0, tau=1.0, warmup=1, step_scale=1000)
        for t in [1, 500, 1000]:
            assert g(t) == 1.0

    def test_decay_matches_sample_code(self):
        """Verify ramp matches sample_code.py: rho * exp(-tau * log((t - offset) / step_scale))."""
        g = GainSequence("ramp", rho=1.0, tau=1.0, warmup=1, step_scale=1000)
        t = 5000
        expected = 1.0 * math.exp(-1.0 * math.log((t - 0) / 1000))
        assert abs(g(t) - expected) < 1e-12

    def test_multi_warmup(self):
        g = GainSequence("ramp", rho=2.0, tau=0.5, warmup=3, step_scale=500)
        # warmup phase lasts 3*500 = 1500 iters
        assert g(1) == 2.0
        assert g(1500) == 2.0
        # After warmup, decay kicks in
        assert g(1501) < 2.0


class TestCustom:
    def test_callable(self):
        g = GainSequence(lambda t: 1.0 / t**2)
        assert g(1) == 1.0
        assert g(10) == 0.01

    def test_as_tensor(self):
        g = GainSequence(lambda t: float(t))
        tensor = g.as_tensor(5)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestAsTensor:
    def test_shape_and_device(self):
        g = GainSequence("1/t", t0=10)
        t = g.as_tensor(100)
        assert t.shape == (100,)
        assert t.device == torch.device("cpu")

    def test_values_match_call(self):
        g = GainSequence("1/t", t0=10)
        t = g.as_tensor(50)
        for i in range(50):
            assert abs(t[i].item() - g(i + 1)) < 1e-12


class TestInvalidSchedule:
    def test_unknown_string(self):
        try:
            GainSequence("unknown")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
