# FUMEplot
Facilitated visualisation of Uncertainty in Model Ensemble outputs using plots.

## local usage
For testing try (while in "fumeplot" directory):

<!-- ```
python3 PlotEnsembleLines.py
``` -->
```
python3 PlotNamedSingleByTimestep.py 
```

Uses default, which is "flee".
You can also use "facs" or "homecoming" as an argument.

```
python3 PlotNamedSingleByTimestep.py flee
```
```
python3 PlotNamedSingleByTimestep.py facs
```
```
python3 PlotNamedSingleByTimestep.py homecoming
```

<!-- ```
python3 PlotEnsembleLines.py flee
```

```
python3 PlotEnsembleLines.py homecoming
``` -->

## usage with FabSim3 (fully automated)

FUMEplot acts as an independent plugin to FabSim3, enabling output plotting and analysis for a wide range of its supported codes. To install it, simply run:

```
fabsim localhost install_plugin:FUMEplot
```

Next, to use it, run *any* workflow, and fetch its results to the local machine using 

```
fabsim <machine> fetch_results
```

At this point, FUMEplot can be invoked as wollows:

```
fabsim localhost fplot:<name_of_results_subdirectory>
```

FUMEplot will automatically detect the underlying solver, and construct the visualisations specified for that specific code.

