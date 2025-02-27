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

    update_environment(args)
    
    code="flee"

    #headers, sim_indices, data_indices, loc_names, y_label 
    FUMEheader = ReadHeaders.ReadOutHeaders(f"{env.local_results}/{results_dir}/RUNS", mode=code)

    PlotNamedStocksByTimestep.plotNamedStocksByTimestep(code, f"{env.local_results}/{results_dir}/RUNS", "loc_lines", FUMEheader)
