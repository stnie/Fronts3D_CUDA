# General
This folder contains a container definition file, which allows to build a container to run the provided code.
In this file general information regarding the container, software as well as instructions on how to run the code are provided.

# Background data:
background is unorganized without any subfolders.
All files are named with the following schema:
plYYYYMMDD_HH.nc  
e.g. pl20160923_00.nc  

# Frontal data:
front data is unorganized without any subfolders.
All files are named with the following schema:
frYYYYMMDD_HH.nc
e.g. fr20160923_00.nc
Frontal Data should contain the same timestamps as background data with the same sorted order. Else inputs may be wrongly combined.

# Run From Container
## Sofware used for build
The software listed here is downloaded within the container before the code is built. For information regarding software downloaded during compilation refer to [Third-Party-Licenses](Third-Party-Licenses.md)

The container downloads and installs the following libraries during build.
* [openMPI 4.0.1](https://www.open-mpi.org/)

The following packages are installed via apt-get 
* gcc-12 
* g++-12
* python3-pip
* libssl-dev
* libnetcdf-c++4-dev
* tzdata

The following python libraries are installed via pip
* numpy
* plotly
* pandas
* kaleido

## Build container
Install Apptainer and build the container using
```
apptainer build <targetContainerName> Front3D_Container.def
```
This will create a Container named `<targetContainerName>`

## Run container 
### Run built container
Assuming your Front data is provided as described and stored in a folder `<Path/To/Fronts>`
Assuming your Atmospheric data is provided as described and stored in a folder `<Path/To/Background>`

You can then run the code using:
```
mpirun -n <numberOfProcesses> apptainer run --nv <targetContainerName> <Path/To/Background> <Path/To/Fronts> 
```

The output will be written to a folder /opt/FrontSoftware/cufronts/outputs and /opt/FrontSoftware/cufronts/outputsMPI

You may need an overlay for writing results. In this case call the apptainer as follows:
```
mpirun -n <numberOfProcesses> apptainer run --nv --overlay <Folder/To/Write> <targetContainerName> <Path/To/Background> <Path/To/Fronts> 
```
In this case apptainer will create folders within `<Folder/To/Write>`. The output will then probably be located at: `<Folder/To/Write>/upper/opt/FrontSoftware/cufronts/outputs` and `<Folder/To/Write>/upper/opt/FrontSoftware/cufronts/outputsMPI` respectively
`
### Run built container as shell
Alternatively you can start the container as a shell using the command
```
apptainer shell --nv <targetContainerName>
```
You can now compile and run the code as with your regular shell

# Running without container
If you want to run without container, you need to download and compile the necessary libraries yourself. You can then run the code from the build directory with the following calls. 

## creation of data
```
CuFront <Path/To/Background> <Path/To/Fronts> --out-path <Path/To/Output> --background-file-identifier pl --front-file-identifier fr --write-netCDF --mean-width 5
```
## MPI scalability
```
mpirun -n <numProcesses> CuFrontMPI <Path/To/Background> <Path/To/Fronts> --out-path <Path/To/Output> --background-file-identifier pl --front-file-identifier fr --write-netCDF --mean-width 5
```
# Evaluation scripts
Within the scripts subfolder you can find the evaluation scripts. That should be callable from the container using 
```
apptainer shell <targetContainerName>
```
Within the shell you can call
```
python3 ../scripts/plotTempDiffs.py <Path/To/Output> warm,cold,occ_,occ2,stnry output_data
```