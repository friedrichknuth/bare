# ASP Bundle Adjust Run Evaluation

Python library to evaluate outputs from [NASA Ames Stereo Pipeline](https://ti.arc.nasa.gov/tech/asr/groups/intelligent-robotics/ngt/stereo/) bundle adjust runs on Earth focused cameras. 


### Features
- plot detected interest points over images
- plot match points found between two images
- plot dxdy after bundle adjustment
- plot xyz tsai camera positions before and after bundle adjustment
- plot residuals after bundle adjustment
- plot WV3 image footprint and scanner positions
- plot tsai image footprint and camera positions
- use batch functions to plot multiple products at once

### Examples

See [notebooks](./examples/) for examples on how to use the library.

### Installation from source
```
$ git clone https://github.com/friedrichknuth/bare.git
$ cd ./bare
$ pip install -e .
```

### Contributing

_bare_ contains modular libraries that can accomodate additional quality control products, as well as camera model formats. Currently, the library only accomodates tsai and WV3 camera model inputs.

For contribution guidelines and the author's TODO list, please click [here](./CONTRIBUTING.md).

### Licence
This project is licensed under the terms of the [MIT License](./LICENSE.rst)

### References
NASA Ames Stereo Pipeline [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1345235.svg)](https://doi.org/10.5281/zenodo.1345235)
 
Beyer, Ross A., Oleg Alexandrov, and Scott McMichael. "The Ames Stereo Pipeline: NASA's open source software for deriving and processing terrain data." Earth and Space Science 5.9 (2018): 537-548.

Shean, David E., et al. "An automated, open-source pipeline for mass production of digital elevation models (DEMs) from very-high-resolution commercial stereo satellite imagery." ISPRS Journal of Photogrammetry and Remote Sensing 116 (2016): 101-117.



