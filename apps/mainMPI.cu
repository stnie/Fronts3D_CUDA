#include <vector>
#include <filesystem>
#include "cufronts/IO/fs_helper.hpp"
#include "cufronts/processing/processing_steps.cuh"
#include "cufronts/IO/parserOptions.hpp"
#include "cufronts/IO/commandLineParser.hpp"
#include <mpi.h>

#define VERBOSE

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    int commSize;
    int commRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    auto allOptions = frontIO::setupParser(argc, argv);
    auto IOOpts = allOptions.first;
    auto CalcOpts = allOptions.second;

    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    // Count same node processes
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, commSize, MPI_INFO_NULL, &local_comm);
    int rank_on_node;
    MPI_Comm_rank(local_comm, &rank_on_node);
    int thisProcValid = rank_on_node < deviceCount ? 1 : 0;
    int allProcsValid;
    MPI_Allreduce(&thisProcValid, &allProcsValid, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD); 
    if(allProcsValid == 0){
        if(commRank == 0){
            std::cerr << "Number of MPI processes " << commSize << " exceeds number of available GPUs " << deviceCount << " \n"
            << "Stopping execution!" << std::endl;
        }
        exit(1);
    }
    else if(allProcsValid != 1){
        if(commRank == 0){
            std::cerr << "Some process evaluated an invalid value. Validity was evaluated as " << allProcsValid << " instead of 1 \n"
            << "Stopping execution!" << std::endl;
        }
        exit(1);
    }

    if(commSize > 1){
        CalcOpts.gpuID = commRank%deviceCount;
        IOOpts.commRank = commRank;
        IOOpts.commSize = commSize;
    }
    // There is no guarding, when using more processes than GPUs
    // simply stop execution instead
    if(commRank == 0){
        std::cout << CalcOpts.printOptions() << std::endl;
        std::cout << IOOpts.printOptions() << std::endl;
    }

    std::vector<std::vector<fs::path>> allFiles = frontIO::getAllFiles(IOOpts);
    auto background_files = allFiles[0];
    auto front_files = allFiles[1];

    newMethodPrealloc(background_files, front_files, IOOpts.write_path, CalcOpts);
    
    MPI_Finalize();
}
