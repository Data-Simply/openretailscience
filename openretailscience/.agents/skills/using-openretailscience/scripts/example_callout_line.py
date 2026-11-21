"""Example: a revenue line with trading events called out on it."""

import numpy as np
import pandas as pd
from openretailscience.plots import callout_line

rng = np.random.default_rng(42)
weeks = pd.date_range("2023-01-02", periods=104, freq="W-MON")
christmas = 30_000 * np.exp(-(((np.arange(104) % 52) - 50) ** 2) / 8)
weekly = pd.DataFrame({"week_start": weeks, "revenue": 60_000 + christmas + rng.normal(0, 1_500, 104)})

# One row per callout. The x column must be the same type as the line's x values (real dates
# here, not strings); a date between two rows is anchored on the line between them.
events = pd.DataFrame(
    {
        "week_start": pd.to_datetime(["2023-04-10", "2023-12-18", "2024-07-01"]),
        "event": ["10 Apr 2023", "18 Dec 2023", "1 Jul 2024"],
        "what_happened": ["Easter weekend", "Christmas week sets a revenue record", "Loyalty scheme relaunch"],
    },
)

# Set the size here, not afterwards: the callouts are laid out for the figure as it is now.
# If they cannot all fit without overlapping, plot() raises a ValueError naming the ones left over.
callout_line.plot(
    df=weekly,
    value_col="revenue",
    x_col="week_start",
    callouts=events,
    callout_title_col="event",
    callout_text_col="what_happened",  # give the title column, the text column or both
    eyebrow="Trading",
    title="Weekly revenue and what moved it",
    subtitle="January 2023 to December 2024",
    source_text="Source: OpenRetailScience - 2024",
    figsize=(10, 6),
)
