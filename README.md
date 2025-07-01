# Towards An Automated Assessment of Deviation Desirability

## About
This repository contains the proof-of-concept scripts and results as described in the manuscript <i>Towards An Automated Assessment of Deviation Desirability</i> submitted to the EdbA Workshop, co-located with the ICPM 2025.


## Data
The used data from the BPI Challenge 2019 can be found in the ```/data``` folder. Original source is: https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853/1

### Built with
* ![platform](https://img.shields.io/badge/MacOS--9cf?logo=Apple&style=social)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.9)

## Project Organization

    ├── data                                             <- In this folder, the used data is located
    ├── process_atoms                                    <- This folder contains source code necessary for declarative conformance checking
    ├ ImpactAlignmentOnTime.ipynb                        <- Notebook that contains the application of our work based on  alignment-based deviations and the time dimension
    ├ ImpactDeclarativeOnOutcome.ipynb                   <- Notebook that contains the application of our work based on  declarative constraint violations and the outcome dimension
    ├ ImpactAlignmentOnTime.ipynb                        <- Notebook that contains the application of our work based on  MP-Declare constraint violations and the cost dimension
    ├── README.md                                        <- The top-level README for users of this project.
    └── LICENSE                                          <- License that applies to the source code in this repository.


## How to run

### Locally

To run the app locally/in your host environment, make sure you've installed:

- Python == 3.11
- poetry

Then install the dependencies into a virtual environment:

```sh
poetry shell
```


