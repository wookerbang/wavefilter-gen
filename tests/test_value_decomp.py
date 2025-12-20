import math
import random
import unittest

import torch

from src.data.value_decomp import MDRConfig, mdr_to_value, torch_compose_value, torch_decompose_mdr, value_to_mdr


class ValueDecompTests(unittest.TestCase):
    def test_python_round_trip_precision(self):
        cfg = MDRConfig(decade_min=-15, decade_max=4)
        rng = random.Random(0)
        for _ in range(500):
            exp = rng.randint(cfg.decade_min, cfg.decade_max)
            mant = rng.uniform(1.0, 9.9)
            v = mant * (10.0**exp)
            m_idx, d_idx, res = value_to_mdr(v, cfg=cfg)
            v2 = mdr_to_value(m_idx, d_idx, res, cfg=cfg, mode="precision")
            self.assertTrue(math.isfinite(v2) and v2 > 0)
            self.assertTrue(abs(math.log10(v2) - math.log10(v)) < 1e-6)
            self.assertLessEqual(abs(res), cfg.residual_log10_scale + 1e-12)

    def test_torch_decompose_matches_python(self):
        cfg = MDRConfig(decade_min=-15, decade_max=4)
        vals = torch.tensor([[1.0e-9, 4.7e-12, float("nan"), 0.0]], dtype=torch.float32)
        mant_idx, dec_idx, res, valid = torch_decompose_mdr(vals, cfg=cfg)
        self.assertEqual(mant_idx.shape, vals.shape)
        self.assertEqual(dec_idx.shape, vals.shape)
        self.assertEqual(res.shape, vals.shape)
        self.assertEqual(valid.shape, vals.shape)
        self.assertTrue(valid[0, 0].item())
        self.assertTrue(valid[0, 1].item())
        self.assertFalse(valid[0, 2].item())
        self.assertFalse(valid[0, 3].item())

        # Compare valid positions with Python decomp.
        for j in (0, 1):
            v = float(vals[0, j].item())
            mi, di, r = value_to_mdr(v, cfg=cfg)
            self.assertEqual(int(mant_idx[0, j].item()), mi)
            self.assertEqual(int(dec_idx[0, j].item()), di)
            self.assertTrue(abs(float(res[0, j].item()) - r) < 1e-5)

    def test_compose_standard_ignores_residual(self):
        cfg = MDRConfig(decade_min=-15, decade_max=4)
        mant_idx = torch.tensor([[0, 0]], dtype=torch.long)  # 1.0
        dec_idx = torch.tensor([[15, 15]], dtype=torch.long)  # decade 0
        res = torch.tensor([[0.02, -0.02]], dtype=torch.float32)
        v_std = torch_compose_value(mant_idx, dec_idx, res, cfg=cfg, mode="standard")
        v_prec = torch_compose_value(mant_idx, dec_idx, res, cfg=cfg, mode="precision")
        self.assertTrue(torch.allclose(v_std[0, 0], v_std[0, 1]))
        self.assertFalse(torch.allclose(v_std, v_prec))


if __name__ == "__main__":
    unittest.main()
