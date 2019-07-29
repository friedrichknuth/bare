# ASP Bundle Adjust Run Evaluation Tool

Inputs
- bundle adjust directory
- image directory
- input camera directory

Outputs
- plot of images with interest points
- plot of images with match points
- plot disparity dxdy
- plot showing xy and z camera positions before and after bundle adjustment
- plot of camera triangulation residuals


## TODO
V0
- cleanup examples notebook
- update readme with description and guidelines
- update version release notes

V1
- plot camera foot prints before and after bundle adjust
- plot camera orientations before and after bundle adjust
- create interactive html plots for residuals
- write csv products to temporary folder and don't mess with content in input folders

## DONE

V0
- put organize code into classes
- add pip install
- add environment yml
- plot xy and z cam positions
- plot interest points over images
- plot match points over images
- plot dxdy
- plot residuals


