"""Repository layout regression tests requiring Python standard library only."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestPublicSourceLayout(unittest.TestCase):
    def test_source_tracks(self):
        for path in (
            "research_7b/train_mid20_edge_debiased_mil_v2.py",
            "research_7b/train_mid20_edge_debiased_mil_sharded_fullsd14_v1.py",
            "edge_vitb/models/eviroute.py",
            "edge_vitb/models/backbone.py",
            "edge_vitb/inference/infer_pytorch.py",
            "docs/METHOD.md",
        ):
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())

    def test_core_method_exports(self):
        source = ROOT / "research_7b/train_mid20_edge_debiased_mil_v2.py"
        syntax = ast.parse(source.read_text(encoding="utf-8"))
        declarations = {
            node.name
            for node in syntax.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for symbol in (
            "TopKMILScorer", "apply_dropout_masks", "edge_mass_loss",
            "checkpoint_score", "train_one_epoch", "evaluate_payload",
        ):
            self.assertIn(symbol, declarations)

    def test_readme_has_core_navigation(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("research_7b/train_mid20_edge_debiased_mil_v2.py", readme)
        self.assertIn("edge_vitb/inference/infer_pytorch.py", readme)


if __name__ == "__main__":
    unittest.main()
