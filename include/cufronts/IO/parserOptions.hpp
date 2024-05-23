#pragma once
#include <filesystem>
#include <argparse/argparse.hpp>
#include <sstream>

namespace fs = std::filesystem;

namespace frontIO{
    struct parserOptions{
        static void add_static_arguments(argparse::ArgumentParser& parser){
            parser.add_argument("background-path")
                .help("path to netCDF input folder");
            parser.add_argument("front-path")
                .help("path to frontal data input folder");
            parser.add_argument("--out-path")
                .help("path to where outputs should be written")
                .default_value(".");
            parser.add_argument("--background-file-identifier")
                .help("identifier to filter input background files (e.g. 2016 if only files containing 2016 should be considered)")
                .default_value("pl");
            parser.add_argument("--front-file-identifier")
                .help("identifier to filter input front files (e.g. 2016 if only files containing 2016 should be considered)")
                .default_value("fr");
            parser.add_argument("--data-offset")
                .help("offset in file list, determining where to begin calculation.")
                .default_value(0)
                .scan<'i',int>();
            parser.add_argument("--num-files-to-process")
                .help("number of files to process.")
                .default_value(-1)
                .scan<'i',int>();
        }

        fs::path background_path;
        fs::path front_path;
        fs::path gradient_path = "";
        fs::path write_path;
        std::string bgFileIdentifier="";
        std::string frontFileIdentifier="";
        int data_offset = 0;
        int num_files_to_process = 0;
        // default values. Might be set by MPI if multiprocessing is used.
        int commRank = 0;
        int commSize = 1;

        parserOptions(argparse::ArgumentParser& parser){
            background_path = parser.get<std::string>("background-path");
            front_path = parser.get<std::string>("front-path");
            write_path = parser.get<std::string>("out-path");
            bgFileIdentifier = parser.get<std::string>("--background-file-identifier");
            frontFileIdentifier = parser.get<std::string>("--front-file-identifier");
            data_offset = parser.get<int>("data-offset");
            num_files_to_process = parser.get<int>("num-files-to-process");
        }
        std::string printOptions(){
            std::stringstream ss;
            ss << "background path: " << background_path << "\n";
            ss << "front path: " << front_path << "\n";
            ss << "out path: " << write_path << "\n";
            ss << "bg Identifier: " << bgFileIdentifier << "\n";
            ss << "front Identifier: " << frontFileIdentifier << "\n";
            ss << "begin optimizing from File: " << data_offset << "\n";
            ss << "optimizing number of Files: " << num_files_to_process << "\n";
            ss << "I am rank : " << commRank << "\n";
            ss << "of num ranks: " << commSize << "\n";
            return ss.str();
        }
    };
};