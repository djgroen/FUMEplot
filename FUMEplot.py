try:
    from fabsim.base.fab import *
    from fabsim.VVP import vvp
except ImportError:
    from base.fab import *

from fumeplot import PlotNamedStocksByTimestep,ReadHeaders

# Add local script, blackbox and template path.
add_local_paths("FUMEplot")


@task
def fplot(results_dir, **args):

    code="flee"

    #headers, sim_indices, data_indices, loc_names, y_label 
    FUMEheader = ReadHeaders.ReadOutHeaders(results_dir, mode=code)

    PlotNamedStocksByTimestep.plotNamedStocksByTimestep(code, results_dir, "loc_lines", FUMEheader)
