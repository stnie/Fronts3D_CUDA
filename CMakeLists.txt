cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CuFront VERSION 0.0.1 
        DESCRIPTION "3D Front detection on CUDA"
	LANGUAGES CXX CUDA C)


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    # Set a default build type if none was specified
    set(DEFAULT_BUILD_TYPE "Release")

    set(CMAKE_C_COMPILER $ENV{CC})
    set(CMAKE_CXX_COMPILER $ENV{CXX})
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(cuda_flags -O3 --expt-extended-lambda --expt-relaxed-constexpr -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_86,code=sm_86)

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    include(CTest)

    find_package(Doxygen)
    if(Doxygen_FOUND)
        add_subdirectory(docs)
    else()
        message(STATUS "Doxygen not found, not building docs")
    endif()
endif()


####
#fetch content parts
####

include(FetchContent)


MESSAGE(STATUS "${CMAKE_MODULE_PATH}")


FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG master
)

FetchContent_Declare(
    netcdf-cxx4
    GIT_REPOSITORY https://github.com/Unidata/netcdf-cxx4.git
    GIT_TAG master
)



FetchContent_Declare(
    cudahelpers
    GIT_REPOSITORY https://gitlab.rlp.net/pararch/hpc_helpers.git
    GIT_TAG master
)


FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.10.0
    GIT_PROGRESS TRUE
)


FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm.git
    GIT_TAG v23.02.00
    GIT_PROGRESS TRUE
    #CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)


FetchContent_Declare(
    cccl
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG v2.2.0
    GIT_PROGRESS TRUE
)

# We will not build rmm, but rather just include the headers
FetchContent_GetProperties(rmm)
FetchContent_Populate(rmm)


FetchContent_MakeAvailable(cudahelpers argparse spdlog cccl netcdf-cxx4)


###################################################################################################
# - find packages we depend on --------------------------------------------------------------------

find_package(CUDAToolkit 12.1 REQUIRED)

find_package(MPI)

find_package(netCDF REQUIRED)
IF(netCDF_FOUND)
    MESSAGE("netCDF was found")
ENDIF()


########################################
# - create targets

# The compiled library code
add_subdirectory(src)

# The executable code
add_subdirectory(apps)
