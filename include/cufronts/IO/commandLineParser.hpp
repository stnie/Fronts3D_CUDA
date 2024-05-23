#include <argparse/argparse.hpp>
#include <cufronts/IO/parserOptions.hpp>
#include <cufronts/processing/util/options.hpp>


namespace frontIO{
    std::pair<frontIO::parserOptions, fronts::options> setupParser(int argc, char * argv[]){
        argparse::ArgumentParser parser("3DFronts Cuda");
        frontIO::parserOptions::add_static_arguments(parser);
        fronts::options::add_static_arguments(parser);

        try{
            parser.parse_args(argc, argv);
        }
        catch (const std::exception& err){
            std::cerr << err.what() << std::endl;
            std::cerr << parser;
            exit(1);
        }

        frontIO::parserOptions pOpt(parser);
        fronts::options cOpt(parser);
        return std::pair<frontIO::parserOptions, fronts::options>{std::move(pOpt), std::move(cOpt)};
    }
}