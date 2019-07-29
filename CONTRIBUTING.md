# Contribution Guidelines

Functions should be organized and added to the following classes, following [PEP8](http://www.python.org/dev/peps/pep-0008/) conventions.



### Libraries

#### bare/core.core.py 
Functions to process bundle adjust outputs

#### bare/common.common.py 
Basic utilities and tools.

#### bare/geospatial.geospatial.py
Geospatial data wrangling utilities and tools.

#### bare/plot.plot.py 
Functions to plot bundle adjust products.


### TODO
V0.1
- change to generic example from ASP documentation
- add Binder integration
- improve / add docstrings
- add release tag

V0.2
- make use of contextily optional
- plot camera foot prints before and after bundle adjust
- plot camera orientations before and after bundle adjust
- create interactive html plots for residuals
- write csv products to temporary folder, instead of adding content in ASP bundle adjust input folders
- compile output plots in pdf report

V0.3
- organize into classes if need for instance properties arises

