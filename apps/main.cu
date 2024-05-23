#include <vector>
#include <filesystem>
#include "cufronts/IO/fs_helper.hpp"
#include "cufronts/processing/processing_steps.cuh"
#include "cufronts/IO/parserOptions.hpp"
#include "cufronts/IO/commandLineParser.hpp"

#define VERBOSE

int main(int argc, char* argv[]){
    auto allOptions = frontIO::setupParser(argc, argv);
    auto IOOpts = allOptions.first;
    auto CalcOpts = allOptions.second;

    std::cout << CalcOpts.printOptions() << std::endl;
    std::cout << IOOpts.printOptions() << std::endl;

    std::vector<std::vector<fs::path>> allFiles = frontIO::getAllFiles(IOOpts);
    auto background_files = allFiles[0];
    auto front_files = allFiles[1];
    newMethodPrealloc(background_files, front_files, IOOpts.write_path, CalcOpts);
}
