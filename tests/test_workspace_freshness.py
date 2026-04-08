from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WorkspaceFreshnessTests(unittest.TestCase):
    def test_old_paper_package_is_gone(self) -> None:
        self.assertFalse((ROOT / "paper").exists())
        self.assertTrue((ROOT / "paper_rendezvous").exists())

    def test_old_plotter_is_gone(self) -> None:
        self.assertFalse((ROOT / "visualizations" / "plot_paper_figures.py").exists())
        self.assertTrue((ROOT / "visualizations" / "plot_rendezvous_figures.py").exists())


if __name__ == "__main__":
    unittest.main()
