try:
    from fabsim.base.fab import *
    from fabsim.VVP import vvp
except ImportError:
    from base.fab import *

from fumeplot import *

# Add local script, blackbox and template path.
add_local_paths("FUMEplot")


@task
def fplot(results_dir, **args):
    PlotNamedStocksByTimestep("flee", results_dir, "loc_lines")
