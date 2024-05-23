#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <cufronts/types/math_helper.cuh>
#include <argparse/argparse.hpp>
#include <sstream>
#include <fstream>

namespace fronts{
    struct options{
        static void add_static_arguments(argparse::ArgumentParser& parser){
            parser.add_argument("--latrange")
                .help("latitude range to consider (begin, end, step)")
                .default_value(std::vector<float>{90,0,-0.25})
                .nargs(3)
                .scan<'g',float>();
            parser.add_argument("--lonrange")
                .help("longitude range to consider (begin, end, step)")
                .default_value(std::vector<float>{-90,50,0.25})
                .nargs(3)
                .scan<'g',float>();
            parser.add_argument("--levels")
                .help("all levels (in hPa) to consider")
                .default_value(std::vector<float>{500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000})
                .nargs(1,16)
                .scan<'g',float>();
            parser.add_argument("--rotation-samples")
                .help("how many rotation oversamples should be done")
                .default_value(1)
                .scan<'i',int>();
            parser.add_argument("--interpolation-points")
                .help("how many points should be interpolated between two vertices")
                .default_value(16)
                .scan<'i',int>();
            parser.add_argument("--disable-bezier")
                .help("disbale additional bezier interpolation")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--stepsize")
                .help("distance between two points on a cross section in km")
                .default_value(20.0f)
                .scan<'g',float>();
            parser.add_argument("--orientation-check-stepsize")
                .help("distance between two points when checking for optimal orientation of normal in km")
                .default_value(20.0f)
                .scan<'g',float>();
            parser.add_argument("--mean-width")
                .help("width of mean calculation during scoring")
                .default_value(5)
                .scan<'i',int>();
            parser.add_argument("--optimization-window-size")
                .help("width of optimization window")
                .default_value(16)
                .scan<'i',int>();
            parser.add_argument("--disable-adjust-orientation")
                .help("perform no orientation correction")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--disable-optimize-paths")
                .help("do not perform path optimization")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--write-cross-sections")
                .help("enable output calculated cross sections")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--write-segment-sizes")
                .help("enable output calculated segment sizes")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--disable-write-directions")
                .help("disable output calculated directions per front point")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--disable-write-scores")
                .help("disable output calculated scores per front point")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--disable-write-seps")
                .help("disable output calculated separation index per front point")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--disable-write-coords")
                .help("disable output calculated coordinates per front point")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--write-netCDF")
                .help("enable output netCDF File (slower due to no parallel IO)")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--verbose-timer")
                .help("print more granular timings")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--verbose-timer-calc")
                .help("print more granular timings")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--verbose-timer-read")
                .help("print more granular timings")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--verbose-timer-write")
                .help("print more granular timings")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--verbose-progress")
                .help("print progress information")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--random-eval")
                .help("randomly pick frontal points on each cross section instead of optimizing")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--static-eval")
                .help("always pick frontal points at the center of each cross section instead of optimizing")
                .default_value(false)
                .implicit_value(true);
            parser.add_argument("--gpuID")
                .help("select which gpu to use. Only used during single processing")
                .default_value(0)
                .scan<'i',int>();
            parser.add_argument("--initial-coord-stddev")
                .help("noise added to coordinates for evaluation purposes only")
                .default_value(0.0f)
                .scan<'g',float>();
            parser.add_argument("--baroclinity")
                .help("use baroclinity evaluation")
                .default_value(false)
                .implicit_value(true);
        }

        // Fixed parameter (channel to process)
        const std::vector<std::string> out_chnls = {"warm", "cold", "occ", "occ2", "stnry"};
        const std::vector<bool> reversal = {false, false, false, true, false};
        // Algorithm currently only works on theta_e and only one conversion is implemented
        const int variables = 1;
        std::vector<std::string> netCDF_vars = {"t","q","r"};
        const std::vector<std::string> netCDF_coords = {"level","latitude","longitude"};
        // Currently Fixed. May become an option in a future version
        const std::vector<std::string> in_chnls = {"warm", "cold", "occ", "stnry"};


        // fixed due to no thread safety of netcdf
        const int num_reader = 1;
        // fixed width as several kernel sizes rely on this
        const int cross_section_width = 256;

        // adjustable parameter for the region
        val3<float> latrange = {90, 0 ,-0.25};
        val3<float> lonrange = {-90, 50, 0.25};
        std::vector<float> levels = {500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000};
        // implicitly set parameters for reading
        int level = levels.size();
        std::vector<float> start_coords = {levels[0], latrange.x, lonrange.x};
        std::vector<float> end_coords = {levels[level-1], latrange.y, lonrange.y};

        // caluclation parameter
        // additionally rotated samples
        int rotations = 1;
        // interpolations between two vertices
        int interpolations = 16;
        // 1 for linear interpolation only, 2 for linear + bezier interpolation
        int smoothing_steps = 2;
        // distance between two points on a cross section in km
        float step_km = 10;
        // distance between two points on a cross section in km
        float dir_step_km = 10;
        // width of mean calculation during scoring
        int mean_width = 5;
        // width of optimization window
        int optimization_window_size = 16;
        // perform orientation correction
        bool adjust_orientation = true;
        // perform path optimization instead of pixel optimization
        bool optimize_paths = true;
        bool optimize_horizontal_paths = false;
        bool filter_invalid_samples = false;
        
        // output options
        bool write_cross_sections = false;
        bool write_segment_sizes = true;
        bool write_dirs = true;
        bool write_scores = true;
        bool write_seps = true;
        bool write_coords = true;
        bool write_netCDF = false;

        bool verbose_timer = false;
        bool verbose_timer_calc = false;
        bool verbose_timer_read = false;
        bool verbose_timer_write = false;

        bool verbose_progress = false;
        bool random_eval=false;
        bool static_eval=false;

        int gpuID = 0;

        // some noise for evaluation purposes added to input coordinates
        float initial_coord_stddev = 0;
        // produce cross sections of z for evaluation purposes
        bool baroclinity_eval = false;

        

        options(argparse::ArgumentParser& clo){
            auto inLatrange = clo.get<std::vector<float>>("--latrange");
            auto inLonrange = clo.get<std::vector<float>>("--lonrange");
            auto inLevel = clo.get<std::vector<float>>("--levels");
            val3<float> myLatrange{inLatrange[0],inLatrange[1],inLatrange[2]};
            val3<float> myLonrange{inLonrange[0],inLonrange[1],inLonrange[2]};
            setGridRanges(myLatrange, myLonrange, inLevel);
            

            // additionally rotated samples
            rotations = clo.get<int>("--rotation-samples");
            // interpolations between two vertices
            interpolations = clo.get<int>("--interpolation-points");
            // 1 for linear interpolation only, 2 for linear + bezier interpolation
            smoothing_steps = 2-1*clo.get<bool>("disable-bezier");
            // distance between two points on a cross section in km
            step_km = clo.get<float>("--stepsize");
            // distance between two points on a cross section in km
            dir_step_km = clo.get<float>("--orientation-check-stepsize");
            // width of mean calculation during scoring
            mean_width = clo.get<int>("--mean-width");
            // width of optimization window
            optimization_window_size = clo.get<int>("--optimization-window-size");
            // perform orientation correction
            adjust_orientation = !clo.get<bool>("--disable-adjust-orientation");
            // perform path optimization instead of pixel optimization
            optimize_paths = !clo.get<bool>("--disable-optimize-paths");
            
            // output options
            write_cross_sections = clo.get<bool>("write-cross-sections");
            write_segment_sizes = clo.get<bool>("write-segment-sizes");
            write_dirs = !clo.get<bool>("disable-write-directions");
            write_scores = !clo.get<bool>("disable-write-scores");
            write_seps = !clo.get<bool>("disable-write-seps");
            write_coords = !clo.get<bool>("disable-write-coords");
            write_netCDF = clo.get<bool>("write-netCDF");

            verbose_timer = clo.get<bool>("verbose-timer");
            verbose_timer_calc = verbose_timer || clo.get<bool>("verbose-timer-calc");
            verbose_timer_read = verbose_timer || clo.get<bool>("verbose-timer-read");
            verbose_timer_write = verbose_timer || clo.get<bool>("verbose-timer-write");
	    
            verbose_progress = clo.get<bool>("verbose-progress");
            static_eval = clo.get<bool>("static-eval");
            random_eval = clo.get<bool>("random-eval");

            gpuID = clo.get<int>("gpuID");

            initial_coord_stddev = clo.get<float>("--initial-coord-stddev");

            baroclinity_eval = clo.get<bool>("--baroclinity");
	        if(baroclinity_eval) netCDF_vars.push_back("z");
            
        }




        void setGridRanges(val3<float> latrange, val3<float> lonrange, std::vector<float> levels){
            this->latrange = latrange;
            this->lonrange = lonrange;
            this->levels = levels;
            level = levels.size();
            start_coords = {levels[0], latrange.x, lonrange.x};
            end_coords = {levels[level-1], latrange.y, lonrange.y};
        }
        
        int dimlat(){
            return abs((latrange.x-latrange.y)/latrange.z)+1;
        }
        int dimlon(){
            return abs((lonrange.x-lonrange.y)/lonrange.z)+1;
        }

        std::string printOptions(){
            std::stringstream ss;
            ss << "latrange: " << latrange.x << ", " << latrange.y << ", " << latrange.z << "\n";
            ss << "lonrange: " << lonrange.x << ", " << lonrange.y << ", " << lonrange.z << "\n";
            ss << "levels: ";
            for(auto& l: levels)
                ss << l << ", ";
            ss << "\n";
            ss << "numLevel = " << level << "\n";
            ss << "start coords: ";
            for(auto& l: start_coords)
                ss << l << ", ";
            ss << "\n";
            ss << "end coords: ";
            for(auto& l: end_coords)
                ss << l << ", ";
            ss << "\n";
            ss << "rotations: " << rotations << "\n";
            ss << "interpolations: " << interpolations << "\n";
            ss << "smoothing steps: " << smoothing_steps << "\n";
            ss << "step km: " << step_km << "\n";
            ss << "mean width: " << mean_width << "\n";
            ss << std::boolalpha;
            ss << "adjust orientation: " << adjust_orientation << "\n";
            ss << "optimize paths: " << optimize_paths << "\n";
            ss << "write cross sections: " << write_cross_sections << "\n";
            ss << "write segment sizes: " << write_segment_sizes << "\n";
            ss << "write dirs: " << write_dirs << "\n";
            ss << "write scores: " << write_scores << "\n";
            ss << "write seps: " << write_seps << "\n";
            ss << "write coords: " << write_coords << "\n";
            ss << "write netCDF: " << write_netCDF << "\n";
            ss << "verbose timer: " << verbose_timer << "\n";
            ss << "verbose progress: " << verbose_progress << "\n";
            ss << "static eval: " << static_eval << "\n";
            ss << "random eval: " << random_eval << "\n";
            ss << "using GPU: " << gpuID << "\n";
            ss << std::noboolalpha;
            return ss.str();
        }



    };
}
