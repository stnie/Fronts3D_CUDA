# CuFronts

## Usage
CuFronts is a GPU based tool for the expansion of two dimensional frontal lines to fully three dimensional data. 

## Input
CuFronts needs two different types of input. 
ERA5 data on pressure levels. Containing at least the following variables
latitude, longitude, plev, t, q, r 

Frontal Data provided as polylines in a format as [zenodo](https://doi.org/10.5281/zenodo.11241530)


## Preconditions
### Your system should have the following preinstalled 
HDF5 Library with parallel IO support (was tested on version 1.12.1)
netCDF C Library with parallel IO supoort (was tested on version 4.8.1)
GCC Compiler or similar (was tested on version 11.2.0)
CMAKE (at least version 3.18)

### For multiprocessing support
OpenMPI (was tested on version 4.1.1)
CUDAToolkit (at least version 12.1)


## Downloaded during build
These Libraries need not be preinstalled in your system, but are fetched from their respective repositories 
During build cmake will try to fetch several other libraries from github and gitlab. None of these libraries are distributed within this repository. Checkout their licenses from their respecitve repositories.
Make sure to have access to these systems. 

downloaded libraries:
[argparse](https://github.com/p-ranav/argparse.git)  (github, [MIT License](https://github.com/p-ranav/argparse/blob/master/LICENSE))
[netcdf-cxx4](https://github.com/Unidata/netcdf-cxx4.git)  (github, [Copyright](https://github.com/Unidata/netcdf-cxx4/blob/master/COPYRIGHT))
[hpc helpers](https://gitlab.rlp.net/pararch/hpc_helpers.git) (gitlab, [MIT License](https://gitlab.rlp.net/pararch/hpc_helpers/-/blob/namespaces/LICENSE))
[spdlog](https://github.com/gabime/spdlog.git) (github, [MIT License](https://github.com/gabime/spdlog/blob/v1.x/LICENSE))
[rmm](https://github.com/rapidsai/rmm.git) (github, [License] (https://github.com/rapidsai/rmm/blob/branch-24.02/LICENSE))
[cccl](https://github.com/NVIDIA/cccl.git) (github, [License](https://github.com/NVIDIA/cccl/blob/main/LICENSE))


## build
This tool can be built using CMake from the root directory

```
mkdir build
cd build
cmake ..
make
```

### If compilation succeeds the executable should be located at:
cufronts/build/apps/CuFront 
### if MPI version is built
cufronts/build/apps/CuFrontMPI  

## execute
### To run the code with default arguments simply call
```
CuFront <path/to/atmospheric/NetCDF/data> <path/to/front/data> 
```
### To create netCDF output call 
```
CuFront <path/to/atmospheric/NetCDF/data> <path/to/front/data> --write-netCDF 
```
### Filter Files
The code recursively goes through the input directories and searches for all files starting with a given identifier. Default for NetCDF data is "pl". Default for front data is "fr". These values can be adjusted with the --background-file-identifier  or --front-file-identifier property. 

The following call for example only considers NetCDF files that start with background201609 and front files starting with fronts201609. If file names are chosen correspondingly this could be used to filter certain dates. 

```
CuFront <path/to/atmospheric/NetCDF/data> <path/to/front/data> --write-netCDF --background-file-identifier background201609 --front-file-identifier fronts201609
```

## Important: 
The Algorithm does not check correspondence between background and front files. All found files are sorted and stored in a vector, which is consecutively processed. Ensure that sorting of netCDF-file paths and front-file paths automatically create correspondence.  

## Container:
Information about a container to execute the code can be found in the evaluation subfolder [README](evaluations/README.md)

