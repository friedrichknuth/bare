# Contribution Guidelines

Functions should be organized and added to the following classes, following [PEP8](http://www.python.org/dev/peps/pep-0008/) conventions.



### Libraries and Classes

#### bare/bare.py
`Core`
Functions to process bundle adjust outputs

`Plot`
Functions to plot bundle adjust products.

#### bare/common.py
`Common`  
Basic utilities and tools.


#### bare/geospatial.py
`Geospatial`  
Geospatial data wrangling utilities and tools.






### TODO
V0.1
- change to generic example from ASP documentation
- improve / add docstrings
- add release tag

V0.2
- plot camera foot prints before and after bundle adjust
- plot camera orientations before and after bundle adjust
- create interactive html plots for residuals
- write csv products to temporary folder, instead of adding content in ASP bundle adjust input folders
- compile output plots in pdf report

