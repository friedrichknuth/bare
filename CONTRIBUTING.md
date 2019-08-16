# Contribution Guidelines

Functions should be organized and added to the following classes, following [PEP8](http://www.python.org/dev/peps/pep-0008/) conventions.

### Libraries

#### bare/batch.batch.py 
Functions to plot multiple bundle adjust products at once. 

#### bare/core.core.py 
Functions to process bundle adjust outputs.

#### bare/geospatial.geospatial.py
Geospatial data wrangling functions.

#### bare/io.io.py 
Basic io functions.

#### bare/io.io.py 
Product specific plotting functions.

#### bare/plot.plot.py 
Functions to plot bundle adjust products.

#### bare/utils.utils.py 
Wrappers around external tools.

### TODO
V0.1
- change to generic example from ASP documentation
- add Binder integration
- improve / add docstrings
- add release tag

V0.2
- create interactive html plots for residuals and cameras in 3D
- write csv products to temporary folder, instead of adding content in ASP bundle adjust input folders
- compile output plots in pdf report

V0.3
- organize into classes with ctx as conditional property of plots, for example

