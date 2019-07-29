# Contribution Guidelines

Functions should be organized and added to the following classes, following [PEP8](http://www.python.org/dev/peps/pep-0008/) conventions.


### Classes
#### BundleAdjustRunEvaluation

File: `bare/bare.py`

Main functions to plot and evaluate bundle adjust runs. Designed specifically to handle ASP outputs.

#### Core

File: `bare/core.py`

Basic os utilities and tools. Not ASP specific.

#### Geospatial

File: `bare/geospatial.py`

Basic geospatial data qrangling utilities and tools. Not ASP specific.




### TODO
V0.1
- improve / add docstrings
- add release tag

V0.2
- plot camera foot prints before and after bundle adjust
- plot camera orientations before and after bundle adjust
- create interactive html plots for residuals
- write csv products to temporary folder, instead of adding content in ASP bundle adjust input folders
- compile output plots in pdf report

