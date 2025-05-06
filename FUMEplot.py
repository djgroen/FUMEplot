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

    runsdir = f"{env.local_results}/{results_dir}/RUNS"

    if results_dir.endswith("_replica_"):
        #replica mode
        runsdir = f"{env.local_results}/{results_dir}1"
        env_file_name = f"{runsdir}/env.yml"
        with open(env_file_name, 'r') as file:
            # Read each line in the file
            for line in file:
                if line.startswith("job_config_path_local:"):
                    return line

            print(f"ERROR: no variable job_config_path_local found in {env_file_name}.")
            sys.exit()

    for name in os.listdir(runsdir):
        env_file_name = f"{runsdir}/{name}/env.yml"
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
    file_loc = os.path.dirname(__file__)

    if "FabFlee" in config_path:
        code="flee"
    if "FabHomecoming" in config_path:
        code="homecoming"
    if "FabCovid19" in config_path:
        code="facs"

    config = {}
    with open(f"{file_loc}/{code}.yml") as config_stream:
        try:
            config = yaml.safe_load(config_stream)
        except yaml.YAMLError as exc:
            print(exc, file=sys.stderr)
            sys.exit()

    print(config)

    outdirs = ReadHeaders.GetOutDirs(f"{env.local_results}/{results_dir}")

    #headers, sim_indices, data_indices, loc_names, y_label 
    FUMEheader = ReadHeaders.ReadOutHeaders(outdirs, mode=code)

    if "NamedSingleByTimestep" in config:
        if code == "homecoming":
            FUMEmovelogheader = ReadHeaders.ReadMovelogHeaders(outdirs, mode=code)
            for m in config["NamedSingleByTimestep"]["modes"]:
                PlotNamedSingleByTimestep.plotNamedSingleByTimestep(code, outdirs, m, FUMEmovelogheader, filters=config["NamedSingleByTimestep"]["filters"], disaggregator=config["NamedSingleByTimestep"]["disaggregator"], primary_filter_column=config["NamedSingleByTimestep"]["primary_filter_column"], primary_filter_value=config["NamedSingleByTimestep"]["primary_filter_value"])
    
    if "NamedStocksByTimestep" in config:
        for m in config["NamedStocksByTimestep"]["modes"]:
            PlotNamedStocksByTimestep.plotNamedStocksByTimestep(code, outdirs, m, FUMEheader)
