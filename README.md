# ASP Bundle Adjust Evaluation Tool
What do I want this library to do?

- Have the ability to import it as a library
- Have a function that will take in the bundle adjust directory, as well as a directory with images, then write out images that show the distribution of match points and plots

Inputs
- bundle adjust directory
- image directory
- input camera directory

Outputs
- plot of images with interest points
- plot of images with match points
- plot dxdy
- plot showing xy and z camera positions before and after bundle adjustment
- plot residuals


## TODO
V0
- cleanup examples notebook
- put code into classes
- update readme with description and guidelines
- update version release notes

V1
- plot camera foot prints before and after bundle adjust
- plot camera orientations before and after bundle adjust
- create interactive html plots for residuals
- write csv products to temporary folder and don't mess with content in input folders

## DONE

V0
- add pip install
- add environment yml
- plot xy and z cam positions
- plot interest points over images
- plot match points over images
- plot dxdy
- plot residuals


