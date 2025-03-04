try:
    from fabsim.base.fab import *
    from fabsim.VVP import vvp
except ImportError:
    from base.fab import *

try:
    from fabsim.plugins.FUMEplot.fumeplot import PlotNamedStocksByTimestep,PlotNamedSingleByTimestep,ReadHeaders
except:
    from plugins.FUMEplot.fumeplot import PlotNamedStocksByTimestep,PlotNamedSingleByTimestep,ReadHeaders


import sys

# Add local script, blackbox and template path.
add_local_paths("FUMEplot")

def load_config_path(results_dir, **args):
    update_environment(args)

    for name in os.listdir(f"{env.local_results}/{results_dir}/RUNS"):
        env_file_name = f"{env.local_results}/{results_dir}/RUNS/{name}/env.yml"
        with open(env_file_name, 'r') as file:
            # Read each line in the file
            for line in file:
                if line.startswith("job_config_path_local:"):
                    return line

            print(f"ERROR: no variable job_config_path_local found in {env_file_name}.")
            sys.exit()
                


@task
def fplot(results_dir, **args):

    update_environment(args)
   
    code="NOT_DETECTED"
    config_path = load_config_path(results_dir)

    if "FabFlee" in config_path:
        code="flee"
    if "FabHomecoming" in config_path:
        code="homecoming"
    if "FabCovid19" in config_path:
        code="facs"

    outdir = f"{env.local_results}/{results_dir}/RUNS"

    #headers, sim_indices, data_indices, loc_names, y_label 
    FUMEheader = ReadHeaders.ReadOutHeaders(outdir, mode=code)


    if code == "homecoming":
        FUMEheader = ReadHeaders.ReadMovelogHeaders(outdir, mode=code)
        PlotNamedSingleByTimestep.plotNamedSingleByTimestep(code, outdir, "loc_lines", FUMEheader)
    else:
        PlotNamedStocksByTimestep.plotNamedStocksByTimestep(code, outdir, "loc_lines", FUMEheader)
