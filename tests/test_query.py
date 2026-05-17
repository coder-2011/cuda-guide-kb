#!/usr/bin/env python3
"""Smoke tests for the CUDA guide query index."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_query(question: str) -> str:
    command = [sys.executable, str(ROOT / "scripts" / "query.py"), question, "--top-k", "5"]
    return subprocess.check_output(command, cwd=ROOT, text=True)


class QuerySmokeTests(unittest.TestCase):
    def test_broad_memory_performance_query_finds_relevant_context(self) -> None:
        output = run_query("How should I think about CUDA memory performance?")
        lowered = output.lower()
        self.assertIn("page:", lowered)
        self.assertTrue(
            any(term in lowered for term in ("memory", "coalesc", "bandwidth", "shared")),
            output,
        )

    def test_exact_sync_symbol_query_finds_syncthreads(self) -> None:
        output = run_query("What does __syncthreads guarantee?")
        lowered = output.lower()
        self.assertIn("__syncthreads", lowered)
        self.assertIn("thread block synchronization", lowered)
        self.assertIn("page:", lowered)

    def test_occupancy_query_finds_occupancy_context(self) -> None:
        output = run_query("Explain occupancy at a high level")
        lowered = output.lower()
        self.assertIn("occupancy", lowered)
        self.assertTrue(any(term in lowered for term in ("active warps", "resident", "multiprocessor")), output)


if __name__ == "__main__":
    unittest.main()
