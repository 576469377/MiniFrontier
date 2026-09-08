"""Small fixtures write to the test filesystem, separate from training storage."""

import os

import pytest


@pytest.fixture(autouse=True)
def small_test_disk_reserve(monkeypatch):
    if "MINIFRONTIER_MIN_FREE_GIB" not in os.environ:
        monkeypatch.setenv("MINIFRONTIER_MIN_FREE_GIB", "1")
