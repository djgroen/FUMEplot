# FUMEplot
Facilitated visualisation of Uncertainty in Model Ensemble outputs using plots.

## Python version and virtual enviroment
This runs best on Python 3.11 and is recommended to be run in a virtual enviroment. 

```
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv -y
python3.11 -m venv ~/venvs/FUMEplotEnv
source ~/venvs/FUMEplotEnv/bin/activate
```

This creates a virutal enviroment to then install dependencies

## Dependencies

Install dependencies by running:

```
pip install -r requirements.txt
```

## local usage
For testing try (while in "fumeplot" directory):

<!-- ```
python3 PlotEnsembleLines.py
``` -->
```
python3 PlotNamedStocksByTimestep.py 
```

Uses default, which is "flee".
You can also use "facs" or "homecoming" as an argument:

```
python3 PlotNamedStocksByTimestep.py flee
```
```
python3 PlotNamedStocksByTimestep.py facs
```
```
python3 PlotNamedStocksByTimestep.py homecoming
```

<!-- ```
python3 PlotEnsembleLines.py flee
```

```
python3 PlotEnsembleLines.py homecoming
``` -->
This will create plots of quantities evolution with their stadard deviation uncertainties, individual runs trajectories, and histogram animation for each data denomination, as well as a violin plot animation for combined data.


To create a single bar plot with uncertainties for all data denominations run:
```
python PlotNamedSingleByStep.by flee
```

## usage with FabSim3 (fully automated)

FUMEplot acts as an independent plugin to FabSim3, enabling output plotting and analysis for a wide range of its supported codes. To install it, simply run:

```
fabsim localhost install_plugin:FUMEplot
```

Next, to use it, run *any* workflow, and fetch its results to the local machine using 

```
fabsim <machine> fetch_results
```

### Ensemble visualisation

At this point, FUMEplot can be invoked as follows for ensembles:

```
fabsim localhost fplot:<name_of_results_subdirectory>
```

FUMEplot will automatically detect the underlying solver, and construct the visualisations specified for that specific code.

### Replicated runs visualisation

Because replicated runs have a slightly different directory structure, FUMEplot should be used there as follows:

```
fabsim localhost fplot:<base_name_of_replica_run_subdirectory>
```

For example, if your replicas have names ranging from ```flee_archer2_24_replica_1``` to ```flee_archer2_24_replica_9```, then the ```base_name_of_replica_run_subdirectory``` should be set to ```flee_archer2_24_replica_```.


