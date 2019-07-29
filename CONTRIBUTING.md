# Contribution Guidelines

Functions should be organized and added to the following classes, following [PEP8](http://www.python.org/dev/peps/pep-0008/) conventions.



### Libraries and Classes

#### bare/bare.py
`BundleAdjustRunEvaluation`  
Main functions to plot and evaluate bundle adjust runs. Designed specifically to handle ASP outputs.

#### bare/common.py
`Basic`  
Common basic os utilities and tools. Not ASP specific.


#### bare/geospatial.py
`Geospatial`  
Basic geospatial data wrangling utilities and tools. Not ASP specific.






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

