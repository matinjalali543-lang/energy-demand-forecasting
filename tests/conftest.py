"""Shared fixtures, built from the seeded synthetic generator.

The suite never needs the network or the downloaded data, so it runs the same
on a laptop and in CI. Fixtures return fresh objects because several tests
mutate a copy.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features import FeatureSpec, daily_issue_times
from synthetic import make_synthetic_frame


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """120 days from 2024-01-01: spans the March DST change and a few holidays."""
    return make_synthetic_frame(start="2024-01-01", days=120)


@pytest.fixture
def issue_time(synthetic_frame) -> pd.Timestamp:
    """An ordinary mid-window issue time."""
    return daily_issue_times(synthetic_frame, FeatureSpec())[20]
