#pragma once
#include <vector>
#include <filesystem>
#include <iostream>
#include "cufronts/IO/parserOptions.hpp"

namespace fs = std::filesystem;

namespace frontIO{

    std::vector<std::vector<fs::path>> getAllFiles(parserOptions popt){

        
        fs::path background_path = popt.background_path;
        fs::path front_path = popt.front_path;

        const std::string front_ident = popt.frontFileIdentifier; 
        const std::string background_ident = popt.bgFileIdentifier; 
        

        std::vector<fs::path> background_files;
        std::vector<fs::path> front_files;
        

        if(fs::is_directory(background_path)){
            auto background_it = fs::recursive_directory_iterator(background_path);
            auto front_it = fs::recursive_directory_iterator(front_path);
            
            for(auto & f: background_it){
                // if a file is found
                if(f.exists() && !f.is_directory()){
                    // if is a file with our desired name
                    if(f.path().string().find(background_ident) != std::string::npos){
                        background_files.push_back(f.path());
                    }
                }
                else continue;
            }
            for(auto & f: front_it){
                // if a file is found
                if(f.exists() && !f.is_directory()){
                    // if is a file with our desired name
                    if(f.path().string().find(front_ident) != std::string::npos){
                        front_files.push_back(f.path());
                    }
                }
                else continue;
            }
        }
        else if(fs::exists(background_path) && !fs::is_directory(background_path)){
            background_files.push_back(background_path);
            front_files.push_back(front_path);
        }
        else{
            std::cerr << "invalid file name specified! " << background_path <<" is neither fold nor file" << std::endl;
            exit(1);
        }
        
        // sort by name 
        // identifier should be the valid sorting!
        std::sort(background_files.begin(), background_files.end());
        std::sort(front_files.begin(), front_files.end());

        
        int numFilesAvailable = background_files.size();
        int dataOffset = popt.data_offset;
        if(popt.num_files_to_process == -1) popt.num_files_to_process = numFilesAvailable-popt.data_offset;
        int numFilesToProcess = min(numFilesAvailable-popt.data_offset, popt.num_files_to_process);

        if(popt.commSize > 1){
            // split files to process evenly among ranks
            int myCommRank = popt.commRank;
            int myCommSize = popt.commSize;
            int myNumFilesToProcess = (numFilesToProcess+myCommSize-1)/myCommSize;
            int myOffset = myCommRank*myNumFilesToProcess+popt.data_offset;
            if(myCommRank+1 == myCommSize){
                myNumFilesToProcess = numFilesToProcess-myOffset;
            }
            dataOffset = myOffset;
            numFilesToProcess = myNumFilesToProcess;
            printf("rank %i/%i: %lu files: from %i to %i\n", myCommRank, myCommSize, numFilesAvailable, dataOffset, dataOffset+numFilesToProcess);
        }
            
        // subsample in multiprocessing
        std::vector<fs::path> myBackgroundFiles(background_files.begin()+dataOffset, background_files.begin()+dataOffset+numFilesToProcess);
        std::vector<fs::path> myFrontFiles(front_files.begin()+dataOffset, front_files.begin()+dataOffset+numFilesToProcess);
        

        std::vector<std::vector<fs::path>> allFiles;
        allFiles.push_back(myBackgroundFiles);
        allFiles.push_back(myFrontFiles);
        return allFiles;
    }

};