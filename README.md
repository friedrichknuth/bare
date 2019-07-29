# ASP Bundle Adjust Run Evaluation

Python library to evaluate outputs from [NASA Ames Stereo Pipeline](https://ti.arc.nasa.gov/tech/asr/groups/intelligent-robotics/ngt/stereo/) bundle adjust runs. 


### Features

- plot xyz tsai camera positions before and after bundle adjustment
- plot detected interest points over images
- plot match points found between two images
- plot dxdy after bundle adjustment
- plot residuals after bundle adjustment

### Examples

See [notebooks](./examples/) for examples on how to use the library.

### Installation from source
```
$ git clone https://github.com/friedrichknuth/bare.git
$ pip install -e bare
```

### Contributing

_bare_ contains modular classes that can be expanded upon to accomodate additional camera model formats, as well as quality control products. Currently, the library only accomodates tsai camera model inputs.

For contribution guidelines and the current TODO list, please click [here](./CONTRIBUTING.md).

### Licence
MIT License Copyright (c) 2019 Friedrich Knuth

### References
NASA Ames Stereo Pipeline [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1345235.svg)](https://doi.org/10.5281/zenodo.1345235)
 
Beyer, Ross A., Oleg Alexandrov, and Scott McMichael. "The Ames Stereo Pipeline: NASA's open source software for deriving and processing terrain data." Earth and Space Science 5.9 (2018): 537-548.

Shean, David E., et al. "An automated, open-source pipeline for mass production of digital elevation models (DEMs) from very-high-resolution commercial stereo satellite imagery." ISPRS Journal of Photogrammetry and Remote Sensing 116 (2016): 101-117.



