
""" This script generates various plots for named Ratios by Timestep. 
    By Ratio it is a propotion of the total number of agents being plotted, 
    Eg: 1:3 or 20% went to a location and 80% in another out of 100%. """
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
from matplotlib.backends.backend_pdf import PdfPages
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
from ProportionBarcharts import stack_prop_plot, plotStackedBarProportions_AgeFromRaw, plotStackedBarProportions_Age, plotStackedBarProportions_Education, plotStackedBarProportions_Gender, plotStackedBarProportions_Property




# ---------------------------------------------------------------------------
# Legacy-name shim: preserve the original generic function name
# ---------------------------------------------------------------------------

def PlotNamedRatioByTimestep(
    outdirs: Sequence[Union[str, os.PathLike]],
    disaggregator: Union[str, Sequence[str]] = 'age_binned',
    filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
    save_fig: bool = True,
    plot_folder: Union[str, os.PathLike] = "plots",
) -> Optional[pd.DataFrame]:
    """Back-compatible wrapper for the original generic function.

    *disaggregator* may be a string or a sequence (first element used).
    If disaggregator in {"age", "age_binned"}, we auto-bin from raw 'age' column.
    Otherwise we call the generic pipeline directly.
    """
    if isinstance(disaggregator, (list, tuple)) and disaggregator:
        disaggregator = disaggregator[0]

    if disaggregator in ("age", "age_binned"):
        # Use the raw-age path to guarantee a binned column.
        return plotStackedBarProportions_AgeFromRaw(
            outdirs,
            filters=filters,
            save_fig=save_fig,
            plot_folder=plot_folder,
        )

    # route to generic
    return stack_prop_plot(
        outdirs,
        group_col=str(disaggregator),
        filters=filters,
        label_map=None,
        color_map=None,  # caller can recolor later
        pretty_name=str(disaggregator).replace('_', ' ').title(),
        html_basename=f"stacked_bar_{disaggregator}_proportions.html",
        plot_folder=plot_folder,
        save_fig=save_fig,
    )

# ---------------------------------------------------------------------------
# If run as a script: tiny CLI demo (very lightweight)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot 100%-stacked net‑arrival proportions."
    )
    parser.add_argument(
        "outdirs", nargs="+",
        help="One or more directories containing migration.log"
    )
    parser.add_argument(
        "group_col",
        help=("Column name to disaggregate "
              "(e.g., gender, education, age_binned, property_in_ukraine)")
    )
    parser.add_argument(
        "--plot-folder", default="plots",
        help="Output folder for HTML plots (default: ./plots)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write the HTML file to disk"
    )
    args = parser.parse_args()

    stack_prop_plot(
        args.outdirs,
        group_col=args.group_col,
        filters=[],
        save_fig=not args.no_save,
        plot_folder=args.plot_folder,
    )