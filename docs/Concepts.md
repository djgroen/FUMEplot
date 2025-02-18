# Concepts in Vamoplot

## Output types

### Spatial Representation
Describes how quantities or object positions are described in the output files.

* **free**: using x/y/z coordinates.
* **grid**: using coordinates in a Carthesian grid.
* **named**: using names of specific locations in a location graph.

### Object Representation
Describes how quantities or objects are represented themselves in the output files.

* **stocks**: Counts of \# of objects in a given location are provided, without details of the objects themselves.
* **single**: Individual object instances.

### Time Representation
Describes how (or if) time is described in the output files.

* **none**: No explicit time indication.
* **timestep**: Individual time point indicated by an integer.
* **date**: Individual time point indicated by a date.
