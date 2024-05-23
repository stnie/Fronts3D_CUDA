#include <cufronts/processing/processing_steps.cuh>
#include <cufronts/processing/scoring/calculateScores.cuh>
#include <cufronts/processing/scoring/optimizing.cuh>
#include <cufronts/processing/textures/texture_handling.cuh>
#include <cufronts/processing/cross_sections/cross_sections.cuh>
#include <cufronts/processing/util/random.hpp>

template<class T>
void convertPositionsToGridPoints(
    fronts::pinned_host_vector<val2<T>>& locs_h, 
    fronts::pinned_host_vector<T>& dirs_h, 
    frontIO::variableMap<fronts::pinned_host_vector<unsigned char>>& out_buffer,
    frontIO::variableMap<std::vector<T>>& packingParameter,
    std::string chnl, const int num_samples, fronts::options& opt){
        int level = opt.level;
        int dimlat = opt.dimlat();
        int dimlon = opt.dimlon();
        auto locs = cuda::std::mdspan(thrust::raw_pointer_cast(locs_h.data()), level, num_samples);
        auto dirs = cuda::std::mdspan(thrust::raw_pointer_cast(dirs_h.data()), num_samples);
        auto posBuffer = cuda::std::mdspan(thrust::raw_pointer_cast(out_buffer[chnl].data()), level,dimlat,dimlon);
        auto dirBuffer = cuda::std::mdspan(thrust::raw_pointer_cast(out_buffer[chnl+"_dir"].data()), level,dimlat,dimlon);

        for(int h = 0; h<level; ++h){
            for(int i =0; i<num_samples;++i){
                auto lonlat = locs(h,i);
                int latpix = (lonlat.y-opt.latrange.x)/opt.latrange.z;
                int lonpix = (lonlat.x-opt.lonrange.x)/opt.lonrange.z;
                posBuffer(h, latpix, lonpix) = (1-packingParameter[chnl][0])/packingParameter[chnl][1];
                dirBuffer(h, latpix, lonpix) = (dirs(i)-packingParameter[chnl+"_dir"][0])/packingParameter[chnl+"_dir"][1];
            }
        }
    }
template<class T>
void clearPositionsToGridPoints(
    fronts::pinned_host_vector<val2<T>>& locs_h, 
    fronts::pinned_host_vector<T>& dirs_h, 
    frontIO::variableMap<fronts::pinned_host_vector<unsigned char>>& out_buffer,
    frontIO::variableMap<std::vector<T>>& packingParameter,
    std::string chnl, const int num_samples, fronts::options& opt){
        thrust::fill(out_buffer[chnl].begin(), out_buffer[chnl].end(), 0);
        thrust::fill(out_buffer[chnl+"_dir"].begin(), out_buffer[chnl+"_dir"].end(), 0);
    }


void getDiffs(frontIO::frontMap<DTYPE_>& fronts_container, std::vector<std::string> in_chnls, 
    std::vector<fronts::pinned_host_vector<val2<DTYPE_>>>& front_coords, 
    std::vector<fronts::pinned_host_vector<size_t>>& offsets,
    std::vector<fronts::pinned_host_vector<size_t>>& segments,
    std::vector<fronts::pinned_host_vector<size_t>>& sortOffsets,
    int level, int factor){
    size_t chnlIdx = 0;
    for(auto ftype: in_chnls){
        auto fronts = fronts_container[ftype];
	    size_t numberOfCoordinates = 0;
        for(auto& front: fronts){
	        numberOfCoordinates += front.size();
	    }
        
        fronts::pinned_host_vector<val2<DTYPE_>> coordinate(numberOfCoordinates);
        fronts::pinned_host_vector<size_t> localOffsets(fronts.size()+1);
        fronts::pinned_host_vector<size_t> localSegments(fronts.size()+1);
        fronts::pinned_host_vector<size_t> localSortOffsets(fronts.size()*level+1);

        size_t offset = 0;
        size_t segment = 0;
        size_t sortOffset = 0;
        
        localOffsets[0] = 0;
        localSegments[0] = 0;
        localSortOffsets[0] = 0;
        size_t coordIdx = 0;
        size_t offsetIdx = 1;
        for(auto& front: fronts){
            for(auto& coords: front){
                coordinate[coordIdx] = coords;
		        coordIdx+=1;
            }
            offset += front.size();
            segment += front.size()-1;
            localOffsets[offsetIdx]=offset;
            localSegments[offsetIdx]=segment;
            for(int l = 0; l<level; ++l){
                sortOffset += (front.size()-1)*factor;
		        // offsetIdx is used as stride parameter here, however we need to remove the initial offset from the multiplication and add it at the end 
                localSortOffsets[(offsetIdx-1)*level+l+1]=sortOffset;
            }
	        offsetIdx+=1;
        }
        
        front_coords[chnlIdx]=coordinate;
        offsets[chnlIdx] = localOffsets;
        sortOffsets[chnlIdx] = localSortOffsets;
        segments[chnlIdx] = localSegments;
	    chnlIdx+=1;
        if(ftype == "occ"){
            front_coords[chnlIdx]=coordinate;
            offsets[chnlIdx] = localOffsets;
            sortOffsets[chnlIdx] = localSortOffsets;
            segments[chnlIdx] = localSegments;
  	        chnlIdx+=1;
        }
    }
}





void newMethodPrealloc(std::vector<fs::path>& background_files, std::vector<fs::path>& front_files, fs::path output_fold, fronts::options opt){
    std::ofstream write_log;
    std::ofstream read_log;
    std::ofstream calc_log;
    write_log.open(output_fold / "timerWrite.txt");
    read_log.open(output_fold / "timerRead.txt");
    calc_log.open(output_fold / "timerCalc.txt");
    std::vector<std::string> names;

    cudaSetDevice(opt.gpuID);

    auto overall_timer = helpers::GpuTimer(0, "total_data all", opt.gpuID, std::cout);
    
    // read thread timer
    auto read_all_timer = helpers::GpuTimer(0, "total_data read all", opt.gpuID, read_log);
    auto read_file_timer = helpers::GpuTimer(0, "total_file read file", opt.gpuID, read_log);
    auto read_netCDF_timer = helpers::GpuTimer(0, "read read netCDF", opt.gpuID, read_log);
    auto read_front_timer = helpers::GpuTimer(0, "read read front", opt.gpuID, read_log);
    auto read_wait_timer_calc = helpers::GpuTimer(0, "wait waiting for buffer free", opt.gpuID, read_log);
    auto read_wait_timer_netCDF = helpers::GpuTimer(0, "wait waiting for writing netCDF done", opt.gpuID, read_log);
    
    // write thread timer
    auto write_all_timer = helpers::GpuTimer(0, "total_data write all", opt.gpuID, write_log);
    auto write_file_timer = helpers::GpuTimer(0, "total_file write file", opt.gpuID, write_log);
    auto create_netCDF_grid_timer = helpers::GpuTimer(0, "calc create netCDF grid", opt.gpuID, write_log);
    auto write_wait_timer_calc = helpers::GpuTimer(0, "wait waiting for new results", opt.gpuID, write_log);
    auto write_wait_timer_netCDF = helpers::GpuTimer(0, "wait waiting for reading netCDF done", opt.gpuID, write_log);
    auto write_netCDF_timer = helpers::GpuTimer(0, "write writing netCDF file", opt.gpuID, write_log);
    auto write_other_timer = helpers::GpuTimer(0, "write writing other output files", opt.gpuID, write_log);
    auto write_clear_grid_timer = helpers::GpuTimer(0, "calc clearing netCDF grid", opt.gpuID, write_log);

    // calc thread timer 
    auto process_all_timer = helpers::GpuTimer(0, "total_data process all", opt.gpuID, calc_log);
    auto process_file_timer = helpers::GpuTimer(0, "total_file process file", opt.gpuID, calc_log);
    auto copy_read_data_to_gpu_timer = helpers::GpuTimer(0, "copy read data to device", opt.gpuID, calc_log);
    auto calc_convert_timer = helpers::GpuTimer(0, "calc Convert data", opt.gpuID, calc_log);
    auto resize_timer = helpers::GpuTimer(0, "calc resizing", opt.gpuID, calc_log);
    auto copy_timer1 = helpers::GpuTimer(0, "copy copy host2Device per channel", opt.gpuID, calc_log);
    auto copy_timer2 = helpers::GpuTimer(0, "copy copy Device2host per channel", opt.gpuID, calc_log);
    auto calc_wait_timer_read = helpers::GpuTimer(0, "wait waiting for new data", opt.gpuID, calc_log);
    auto calc_wait_timer_write = helpers::GpuTimer(0, "wait waiting for result writable", opt.gpuID, calc_log);
    auto thrust_prefill_timer = helpers::GpuTimer(0, "calc thrust initialization", opt.gpuID, calc_log);
    auto texture_timer = helpers::GpuTimer(0, "copy texture copy and create", opt.gpuID, calc_log);
    auto getDiffs_timer = helpers::GpuTimer(0, "calc getDiff", opt.gpuID, calc_log);
    auto get_pos_and_dir_timer = helpers::GpuTimer(0, "calc GetPosAndDir", opt.gpuID, calc_log);
    auto cross_section_timer = helpers::GpuTimer(0, "calc GetCrossSectionAndFlip", opt.gpuID, calc_log);
    auto adjust_orientation_timer = helpers::GpuTimer(0, "calc AdjustOrientation", opt.gpuID, calc_log);
    auto calc_score_timer = helpers::GpuTimer(0, "calc calculateScore", opt.gpuID, calc_log);
    auto eval_score_timer = helpers::GpuTimer(0, "calc calculateEvalScore", opt.gpuID, calc_log);
    auto optimal_path_timer = helpers::GpuTimer(0, "calc getOptimalPath", opt.gpuID, calc_log);
    auto optimize_timer = helpers::GpuTimer(0, "calc OptimizeFront", opt.gpuID, calc_log);


    overall_timer.start();

    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, opt.gpuID);
    
    int maxThreadsPerSM = props.maxThreadsPerMultiProcessor;
    int maxThreadsInABlock = (maxThreadsPerSM%1024==0) ? 1024 : 512;
    std::cout << "Using a maximum of " << maxThreadsInABlock << "Threads in a block" << std::endl;
    CoordinateNoise noiser;
    noiser.initialize();


    frontIO::BinaryReader reader;
    frontIO::netCDFReader ncReader;
    frontIO::netCDFWriter ncWriter;
    frontIO::CSBReader creader(opt.lonrange.x, opt.lonrange.y, opt.latrange.x, opt.latrange.y, opt.lonrange.z, opt.latrange.z, true);

    // how a 3D array would be accessed (fastest index is last)
    dim3 dataDim(opt.level,opt.dimlat(), opt.dimlon());

    // save variable for estimating writing sizes
    int saveSize = 0;
    int sampleCount = 0;

    /* initialization of vectors for calculation */
    int useExtraBuffer =opt.write_netCDF ? 0:1;
    // 
    std::vector<fronts::pinned_host_vector<DTYPE_>> positions(opt.num_reader);
    std::vector<frontIO::variableMap<fronts::pinned_host_vector<DTYPE_>>> variable_data_buffer(opt.num_reader*(1+useExtraBuffer));
    std::vector<frontIO::variableMap<fronts::pinned_host_vector<PACKTYPE_>>> output_data_buffer(opt.num_reader);
    // purely host based with no copies
    std::vector<frontIO::variableMap<std::vector<DTYPE_>>> packingParameter(opt.num_reader);
    frontIO::variableMap<fronts::pinned_host_vector<DTYPE_>> gridDims;
    for(int i = 0; i< opt.num_reader ; ++i){
        positions[i] = fronts::pinned_host_vector<DTYPE_>(opt.dimlat()*opt.dimlon()*2);
        for(int lat = 0; lat < opt.dimlat(); lat++){
            for(int lon = 0; lon < opt.dimlon(); lon++){
                positions[i][lat*opt.dimlon()+lon] = opt.lonrange.x+opt.lonrange.z*lon;
                positions[i][opt.dimlat()*opt.dimlon()+lat*opt.dimlon()+lon] = opt.latrange.x+opt.latrange.z*lat;
            }
        }
        for(int j = 0; j< (1+useExtraBuffer); ++j){
            variable_data_buffer[i*(1+useExtraBuffer)+j] = frontIO::variableMap<fronts::pinned_host_vector<DTYPE_>>();
            for(auto& variable: opt.netCDF_vars){
                variable_data_buffer[i*(1+useExtraBuffer)+j][variable] = fronts::pinned_host_vector<DTYPE_>(opt.level*opt.dimlat()*opt.dimlon());
            }
        }
        output_data_buffer[i] = frontIO::variableMap<fronts::pinned_host_vector<PACKTYPE_>>();
        packingParameter[i] = frontIO::variableMap<std::vector<DTYPE_>>();
        for(auto& variable: opt.out_chnls){
            output_data_buffer[i][variable] = fronts::pinned_host_vector<PACKTYPE_>(opt.level*opt.dimlat()*opt.dimlon());
            output_data_buffer[i][variable+"_dir"] = fronts::pinned_host_vector<PACKTYPE_>(opt.level*opt.dimlat()*opt.dimlon());
            packingParameter[i][variable] = std::vector<DTYPE_>{0,1};
            packingParameter[i][variable+"_dir"] = std::vector<DTYPE_>{0,(float)(2*3.14/(pow(2,sizeof(PACKTYPE_)*8)-1))};
        }
    }

    gridDims["latitude"] = fronts::pinned_host_vector<DTYPE_>(opt.dimlat());
    for(int i = 0;i<opt.dimlat();++i){
        gridDims["latitude"][i] = opt.latrange.x+i*opt.latrange.z;
    }
    gridDims["longitude"] = fronts::pinned_host_vector<DTYPE_>(opt.dimlon());
    for(int i = 0;i<opt.dimlon();++i){
        gridDims["longitude"][i] = opt.lonrange.x+i*opt.lonrange.z;
    }
    gridDims["level"] = fronts::pinned_host_vector<DTYPE_>(opt.level);
    for(int i = 0;i<opt.level;++i){
        gridDims["level"][i] = opt.levels[i];
    }

    std::vector<frontIO::frontMap<DTYPE_>> front_input_buffer(opt.num_reader*(1+useExtraBuffer));

    std::vector<cudaTextureObject_t> myTexs = createTextures<DTYPE_>(dataDim);
    auto& tex = myTexs[0];
    auto& location_tex = myTexs[1];
    auto& dir_base_tex = myTexs[2];

    cudaStream_t calc_stream, copyH2D_stream, copyD2H_stream;
    cudaStreamCreate(&calc_stream);
    cudaStreamCreate(&copyH2D_stream);
    cudaStreamCreate(&copyD2H_stream);
    
    cudaEvent_t texture_copied;
    cudaEventCreate(&texture_copied);
    cudaEvent_t texture_read;
    cudaEventCreate(&texture_read);
    cudaEvent_t data_processed;
    cudaEventCreate(&data_processed);
    cudaEvent_t results_written_event;
    cudaEventCreate(&results_written_event);
    cudaEvent_t local_results_calculated;
    cudaEventCreate(&local_results_calculated);
    cudaEvent_t local_results_copied;
    cudaEventCreate(&local_results_copied);
    
    // Alloc these with the max
    fronts::device_vector<DTYPE_> levels_d(opt.levels.size(), copyH2D_stream);
    cudaMemcpy(levels_d.data(), opt.levels.data(), opt.levels.size()*sizeof(DTYPE_), cudaMemcpyHostToDevice);
    // buffer data for netcdf conversion on GPU
    fronts::device_vector<DTYPE_> t_d(opt.level*opt.dimlat()*opt.dimlon(), copyH2D_stream);
    fronts::device_vector<DTYPE_> q_d(opt.level*opt.dimlat()*opt.dimlon(), copyH2D_stream);
    fronts::device_vector<DTYPE_> r_d(opt.level*opt.dimlat()*opt.dimlon(), copyH2D_stream);
    fronts::device_vector<DTYPE_> z_d(0, copyH2D_stream);
    if(opt.baroclinity_eval) z_d.resize(opt.level*opt.dimlat()*opt.dimlon(), copyH2D_stream);
    // background data buffer for texture creation
    fronts::device_vector<DTYPE_> data_d(opt.level*opt.dimlat()*opt.dimlon(), copyH2D_stream);
    fronts::device_vector<DTYPE_> in_positions_d(2*opt.dimlat()*opt.dimlon(), copyH2D_stream);

    thrust::copy(rmm::exec_policy_nosync(copyH2D_stream), positions[0].begin(), positions[0].end(), in_positions_d.begin());
    // coordinates per line-vertex
    fronts::device_vector<val2<DTYPE_>> coord_d(16, calc_stream);
    // offsets where a front begins
    fronts::device_vector<size_t> offsets_d(16, calc_stream);
    // num segments until a front begins
    fronts::device_vector<size_t> segments_d(16, calc_stream);

    // indices where each front pixel is located
    fronts::device_vector<val2<DTYPE_>> bases(16, calc_stream);
    // direction of each cross section
    fronts::device_vector<DTYPE_> dirs(16, calc_stream);
    // if direction needs to be flipped
    fronts::device_vector<bool> needs_flip(16, calc_stream);
    
    // shuffle vector for size sorted segments
    fronts::device_vector<size_t> positions_d(16, calc_stream);

    
    fronts::device_vector<size_t> keys_d(16, calc_stream);
    fronts::device_vector<size_t> tgt_keys_d(16, calc_stream);
    fronts::device_vector<size_t> orientations_d(16, calc_stream);
    fronts::device_vector<size_t> mean_orientations_d(16, calc_stream);

    // vector containing cross sections
    fronts::device_vector<DTYPE_> vec(opt.level*256*16, calc_stream);
    // coordinates at each cross section point
    fronts::device_vector<val2<DTYPE_>> loc_vec(opt.level*256*16, calc_stream);
    // scores at each cross section point
    fronts::device_vector<DTYPE_> scores(opt.level*256*16, calc_stream);
    // scores at each cross section point
    fronts::device_vector<DTYPE_> out_scores(opt.level*256*16, calc_stream);
    // ideal separation indices per sample per level
    fronts::device_vector<int> septs(opt.level*16, calc_stream);
    // min Scores per sample per level
    fronts::device_vector<DTYPE_> diffs(opt.level*16, calc_stream);
    // ideal separation coordinates per sample per level
    fronts::device_vector<val2<DTYPE_>> seplocs(opt.level*16, calc_stream);

    // output vectors
    fronts::pinned_host_vector<int> seps_h(septs.size());
    fronts::pinned_host_vector<DTYPE_> diffs_h(diffs.size());
    fronts::pinned_host_vector<val2<DTYPE_>> locs_h(seplocs.size());
    fronts::pinned_host_vector<DTYPE_> dirs_h(dirs.size());
    
    #ifndef USE_ATOMICS
        std::vector<std::mutex> work_m(opt.num_reader*(1+useExtraBuffer)); 
        std::vector<std::mutex> read_m(opt.num_reader*(1+useExtraBuffer)); 
        std::mutex overwrite_m, write_m, netCDFIO_m;
        std::vector<std::condition_variable> work_cv(opt.num_reader*(1+useExtraBuffer));
        std::vector<std::condition_variable> read_cv(opt.num_reader*(1+useExtraBuffer));
        std::condition_variable overwrite_cv, write_cv, netCDFIO_cv;
        // simple async read and compute


        std::vector<bool> can_read(opt.num_reader*(1+useExtraBuffer), true);
        std::vector<bool> can_process(opt.num_reader*(1+useExtraBuffer), false);
        bool can_write = false;
        bool can_overwrite_result = true;
        bool can_access_netCDF = true;
    #endif
    #ifdef USE_ATOMICS
        // initialize as clear
        std::atomic_flag can_read = ATOMIC_FLAG_INIT;
        std::atomic_flag can_process = ATOMIC_FLAG_INIT;
        std::atomic_flag can_write = ATOMIC_FLAG_INIT;
        std::atomic_flag can_overwrite_result = ATOMIC_FLAG_INIT;
        std::atomic_flag can_access_netCDF = ATOMIC_FLAG_INIT;


        // set true
        can_process.test_and_set();
        can_write.test_and_set();
        can_access_netCDF.test_and_set();
    #endif

    // very basic multi reader version
    std::vector<std::future<void>> reader_threads;

    for(int reader_id = 0; reader_id < opt.num_reader; ++reader_id){
        reader_threads.push_back(std::async(std::launch::async, [&, reader_id](){
            cudaSetDevice(opt.gpuID);
            #ifdef USE_ATOMICS
                bool myState = true;
            #endif
            read_all_timer.reset();
            read_all_timer.start();
            for(int file_id = reader_id; file_id < background_files.size(); file_id+=opt.num_reader){
                if(opt.verbose_timer_read){ 
		            read_file_timer.reset();
                    read_file_timer.start();
		        }
		        int reader_id_buffer = reader_id+(file_id%2)*useExtraBuffer;

                fs::path background_filename = background_files[file_id];
                fs::path front_filename = front_files[file_id];
                // read lat/lon into data
                if(opt.verbose_timer_read){ 
                    read_wait_timer_calc.reset();
                    read_wait_timer_calc.start();
	        	}
                // lock if last file has not been processed yet
                #ifndef USE_ATOMICS
                {
                    std::unique_lock lk(read_m[reader_id_buffer]);
                    read_cv[reader_id_buffer].wait(lk, [&]{return can_read[reader_id_buffer];});
                    can_read[reader_id_buffer] = false;
                }
                #else
                    while(myState){
                        can_read.wait(true);
                        myState = can_read.test_and_set();
                    }
                #endif
                if(opt.verbose_timer_read){ 
		            read_wait_timer_calc.print();
		        }
                
                // background Atmospheric data
                if(opt.verbose_timer_read){
                    read_wait_timer_netCDF.reset();
                    read_wait_timer_netCDF.start();
                }

                // read ept data into data
                if(opt.verbose_progress) std::cout << "filename: " <<  background_filename << std::endl;
                if(opt.num_reader>1 || opt.write_netCDF){
                    #ifndef USE_ATOMICS
                    {
                        // restrict netCDF access (threadsafety!)
                        std::unique_lock readNC_lk(netCDFIO_m);
                        netCDFIO_cv.wait(readNC_lk, [&]{return can_access_netCDF;});
                        can_access_netCDF = false;
                    }
                    #else
                        while(myState){
                            can_access_netCDF.wait(true);
                            myState = can_access_netCDF.test_and_set();
                        }
                    #endif
		        }
                if(opt.verbose_timer_read){
		            read_wait_timer_netCDF.print();
                }
                if(opt.verbose_timer_read){
		            read_netCDF_timer.reset();
                    read_netCDF_timer.start();
		        }
                ncReader.readCoordRange(background_filename, opt.netCDF_vars, opt.start_coords, opt.end_coords, variable_data_buffer[reader_id_buffer]);
                if(opt.verbose_timer_read) read_netCDF_timer.print();
                if(opt.num_reader>1 || opt.write_netCDF){
                    #ifndef USE_ATOMICS
                    {
                        // give netCDF access free (threadsafety!)
                        std::unique_lock readNC_lk(netCDFIO_m);
                        can_access_netCDF = true;
                        netCDFIO_cv.notify_one();
                    }
                    #else
                        can_access_netCDF.clear();
                        can_access_netCDF.notify_one();
                    #endif
                }

                // frontal data
                if(opt.verbose_timer_read) read_front_timer.reset();
                if(opt.verbose_timer_read) read_front_timer.start();
                // read the frontal data of current timestamp
                front_input_buffer[reader_id_buffer] = creader.read<DTYPE_>(front_filename);
                if(opt.verbose_timer_read) read_front_timer.print();
                
                #ifndef USE_ATOMICS
                {
                    // inform the worker that some data exists
                    std::unique_lock lk(work_m[reader_id_buffer]);
                    can_process[reader_id_buffer] = true;
                    work_cv[reader_id_buffer].notify_one();
                }
                #else
                    can_process.clear();
                    can_process.notify_one();
                #endif
                if(opt.verbose_timer_read) read_file_timer.print();
            }
            read_all_timer.print();
        }));
    }

    // asynchronous processing of read data => creation of 3D fronts out of 2D
    auto res2 = std::async(std::launch::async, [&](){
        cudaSetDevice(opt.gpuID);
        #ifdef USE_ATOMICS
            bool myState = true;
            bool myWriteState = true;
        #endif
        process_all_timer.start();
        for(int file_id = 0;file_id< background_files.size(); file_id+=opt.num_reader){
            if(opt.verbose_timer_calc) process_file_timer.reset();
            if(opt.verbose_timer_calc) process_file_timer.start();
            for(int reader_id = 0; reader_id<opt.num_reader; ++reader_id){
                if(file_id + reader_id >= background_files.size()) continue;
                int reader_id_buffer = reader_id+(file_id%2)*useExtraBuffer;
                if(opt.verbose_timer_calc){
                    calc_wait_timer_read.reset();
                    calc_wait_timer_read.start();
                }
                #ifndef USE_ATOMICS
                    std::unique_lock work_lk(work_m[reader_id_buffer]);
                    work_cv[reader_id_buffer].wait(work_lk, [&]{return can_process[reader_id_buffer];});
                    can_process[reader_id_buffer] = false;
                #else
                    while(myState){
                        can_process.wait(true);
                        myState = can_process.test_and_set();
                    }
                #endif
                    if(opt.verbose_timer_calc){
                        calc_wait_timer_read.print();
                }
                
                

                cudaStreamWaitEvent(copyH2D_stream, texture_read);
                
                if(opt.verbose_timer_calc){
                    copy_read_data_to_gpu_timer.reset();
                    copy_read_data_to_gpu_timer.start();
                }
                thrust::copy(rmm::exec_policy_nosync(copyH2D_stream), variable_data_buffer[reader_id_buffer]["t"].begin(), variable_data_buffer[reader_id_buffer]["t"].end(), t_d.begin());
                thrust::copy(rmm::exec_policy_nosync(copyH2D_stream), variable_data_buffer[reader_id_buffer]["q"].begin(), variable_data_buffer[reader_id_buffer]["q"].end(), q_d.begin());
                thrust::copy(rmm::exec_policy_nosync(copyH2D_stream), variable_data_buffer[reader_id_buffer]["r"].begin(), variable_data_buffer[reader_id_buffer]["r"].end(), r_d.begin());
                if(opt.verbose_timer_calc) copy_read_data_to_gpu_timer.print();
                if(opt.verbose_timer_calc){
                    calc_convert_timer.reset();
                    calc_convert_timer.start();
                }

                conversion::calculate_equivalent_potential_temperature_grid_gpu(t_d,q_d,r_d,levels_d, data_d, val3<size_t>(dataDim.x, dataDim.y, dataDim.z), copyH2D_stream, maxThreadsInABlock);
                //cudaMemcpyAsync(data_d.data(), data[reader_id].data(), sizeof(typename decltype(data)::value_type::value_type)* data_d.size(), cudaMemcpyHostToDevice, copy_stream);
                if(opt.verbose_timer_calc) calc_convert_timer.print();
                if(opt.verbose_timer_calc){ 
                    texture_timer.reset();
                    texture_timer.start();
                }
                
            
                updateTextures(data_d, in_positions_d, myTexs, dataDim, copyH2D_stream);
                    
                if(opt.baroclinity_eval){
                    // download theta_e 
                    thrust::copy(rmm::exec_policy_nosync(copyH2D_stream), variable_data_buffer[reader_id_buffer]["z"].begin(), variable_data_buffer[reader_id_buffer]["z"].end(), z_d.begin());
                    conversion::calculate_baroclinity_grid_gpu(z_d, data_d, t_d, val3<size_t>(dataDim.x, dataDim.y, dataDim.z), copyH2D_stream);
                        updateTex(t_d.data(), myTexs[0], dataDim, copyH2D_stream);
                    cudaDeviceSynchronize();
                }
                cudaEventRecord(texture_copied, copyH2D_stream);
                if(opt.verbose_timer_calc){
                    texture_timer.print();
                }

                cudaEventSynchronize(data_processed);
                if(opt.verbose_timer_calc){ 
                    getDiffs_timer.reset();
                    getDiffs_timer.start();
                }
                std::vector<fronts::pinned_host_vector<val2<DTYPE_>>> front_coords(opt.out_chnls.size());
                std::vector<fronts::pinned_host_vector<size_t>> offsets(opt.out_chnls.size());
                std::vector<fronts::pinned_host_vector<size_t>> segments(opt.out_chnls.size());
                std::vector<fronts::pinned_host_vector<size_t>> sortOffsets(opt.out_chnls.size());

                getDiffs(front_input_buffer[reader_id_buffer], opt.in_chnls, front_coords, offsets, segments, sortOffsets, opt.level, opt.smoothing_steps*opt.interpolations*opt.rotations);
                if(opt.verbose_timer_calc){
                    getDiffs_timer.print();
                }
                if(abs(opt.initial_coord_stddev)>0){
                    std::cout << front_coords[0][0].x << " " << front_coords[0][0].y << std::endl;
                    std::cout << "adding noise of size " << opt.initial_coord_stddev << std::endl;
                    bool perVertex= opt.initial_coord_stddev < 0;
                            noiser.addNoiseToCoords(front_coords, offsets, abs(opt.initial_coord_stddev), perVertex);
                    std::cout << front_coords[0][0].x << " " << front_coords[0][0].y << std::endl;
                }

                
                #ifndef USE_ATOMICS
                // notify to update the read
                {
                    std::unique_lock read_lk(read_m[reader_id_buffer]);
                    can_read[reader_id_buffer] = true;
                    read_cv[reader_id_buffer].notify_one();
                }
                work_lk.unlock();
                #else

                can_read.clear();
                can_read.notify_one();
                #endif

                if(opt.verbose_timer_calc){
                    resize_timer.reset();
                    resize_timer.start();
                }

                size_t max_points = 0;
                size_t max_fronts = 0;
                size_t max_pairs  = 0;
                for(int chnl = 0; chnl< front_coords.size(); ++chnl){
                    auto num_fronts = offsets[chnl].size()-1;
                    auto cpairs = front_coords[chnl].size();
                    max_pairs = max(cpairs, max_pairs);
                    if(cpairs >= num_fronts)
                        max_points = max(cpairs-num_fronts, max_points);
                    max_fronts = max(num_fronts, max_fronts);
                }
                
                auto max_samples = max_points*opt.smoothing_steps*opt.interpolations*opt.rotations;
                
                // Alloc these with the max
                // coordinates per line-vertex
                coord_d.resize(max_pairs, calc_stream);
                // offsets where a front begins
                offsets_d.resize(max_fronts+1, calc_stream);
                // num segments until a front begins
                segments_d.resize(max_fronts+1, calc_stream);

                // indices where each front pixel is located
                bases.resize(max_samples, calc_stream);
                // direction of each cross section
                dirs.resize(max_samples, calc_stream);
                // if direction needs to be flipped
                needs_flip.resize(max_samples, calc_stream);
                
                // shuffle vector for size sorted segments
                positions_d.resize(max_fronts, calc_stream);

                if(opt.adjust_orientation){
                    keys_d.resize(max_samples, calc_stream);
                    tgt_keys_d.resize(max_fronts, calc_stream);
                    orientations_d.resize(max_samples, calc_stream);
                    mean_orientations_d.resize(max_fronts, calc_stream);
                }


                size_t max_cs_pixel = opt.level*opt.cross_section_width*max_samples;
                // vector containing cross sections
                vec.resize(max_cs_pixel, calc_stream);
                // coordinates at each cross section point
                loc_vec.resize(max_cs_pixel, calc_stream);
                // scores at each cross section point
                scores.resize(max_cs_pixel, calc_stream);
                // scores at each cross section point. These scores are only used for evaluation purposes
                out_scores.resize(max_cs_pixel, calc_stream);
                // ideal separation indices per sample per level
                septs.resize(opt.level*max_samples, calc_stream);
                // optimal scores per sample per level
                diffs.resize(opt.level*max_samples, calc_stream);
                // ideal separation coordinates per sample per level
                seplocs.resize(opt.level*max_samples, calc_stream);

                // output vectors // ensure that previous calculation is finished before overwriting host size 
                cudaEventSynchronize(results_written_event);
                seps_h.resize(septs.size());
                diffs_h.resize(diffs.size());
                locs_h.resize(seplocs.size());
                dirs_h.resize(dirs.size());
                if(opt.verbose_timer_calc){
                    resize_timer.print();
                }
                if(opt.verbose_progress) std::cout << "max samples: " <<  max_samples << std::endl;
                
                
                for(int chnl=0; chnl < opt.out_chnls.size(); ++chnl){
                    if(opt.verbose_timer_calc){
                        for(int i = 0; i< segments[chnl].size()-1; ++i){
                            std::cout << "TIMING: " << segments[chnl][i+1]-segments[chnl][i] << " ms " << "_segment sizes " << opt.out_chnls[chnl] << "_" << std::endl; 
                        }
                        std::cout << "TIMING: " << segments[chnl].size() << " ms " << "_total segment sizes " << opt.out_chnls[chnl] << "_" << std::endl; 
                    }
                    if(opt.write_segment_sizes){
                        std::string infilename = background_files[file_id].filename().string().substr(2,11);
                        reader.write(output_fold / fs::path(std::string(infilename+"_"+opt.out_chnls[chnl]+"_segs.bin")), thrust::raw_pointer_cast(segments[chnl].data()), segments[chnl].size());
                    }
                    
                    // front count for this channel
                    size_t num_fronts = offsets[chnl].size()-1;
                    // sample count for this channel
                    auto num_smp = (front_coords[chnl].size()-num_fronts)*opt.smoothing_steps*opt.interpolations*opt.rotations;

                    // cross section size
                    CsExtent cssize(opt.cross_section_width,opt.level,num_smp,1);
                    // Get The Cross section data for each basis and direction
            
                    dim3 CSThreads(32,4,maxThreadsInABlock/(32*4));
                    dim3 CSBlocks(
                        max(1,(opt.cross_section_width+CSThreads.x-1)/CSThreads.x),
                        max(1,(opt.level+CSThreads.y-1)/CSThreads.y), 
                        max(1ul,(num_smp+CSThreads.z-1)/CSThreads.z));
                    
                    // not very efficient:
                    // fastest, mid, slowest
                    dim3 size(cssize.width, cssize.height, cssize.samples);
                    dim3 mdsize(cssize.samples, cssize.height, cssize.width);

                    // wrap vectors in mdspan for better 3D access
                    // adjust dimensions for 3d vectors
                    fronts::crossSectionGrid<typename decltype(vec)::value_type> grid_vec(thrust::raw_pointer_cast(vec.data()), mdsize.x, mdsize.y,mdsize.z);
                    fronts::crossSectionGrid<typename decltype(loc_vec)::value_type> grid_loc_vec(thrust::raw_pointer_cast(loc_vec.data()), mdsize.x, mdsize.y,mdsize.z);
                    fronts::crossSectionGrid<typename decltype(scores)::value_type> grid_scores(thrust::raw_pointer_cast(scores.data()), mdsize.x, mdsize.y,mdsize.z);
                    fronts::crossSectionGrid<typename decltype(out_scores)::value_type> grid_out_scores(thrust::raw_pointer_cast(out_scores.data()), mdsize.x, mdsize.y,mdsize.z);
                    
                    
                    // adjust dimensions for 2d vectors
                    dim3 size2d(cssize.height, cssize.samples);
                    fronts::outputGrid<typename decltype(septs)::value_type> grid_septs(thrust::raw_pointer_cast(septs.data()), size2d.x, size2d.y);
                    fronts::outputGrid<typename decltype(seplocs)::value_type> grid_seplocs(thrust::raw_pointer_cast(seplocs.data()), size2d.x, size2d.y);
                    fronts::outputGrid<typename decltype(diffs)::value_type> grid_diffs(thrust::raw_pointer_cast(diffs.data()), size2d.x, size2d.y);

                    dim3 ScoringThreads(16,2,16);
                    dim3 ScoringBlocks(
                        max(1,(cssize.width+ScoringThreads.x-1)/ScoringThreads.x),
                        max(1,(cssize.height+ScoringThreads.y-1)/ScoringThreads.y),
                        max(1,(cssize.samples+ScoringThreads.z-1)/ScoringThreads.z));
            
                    int path_optimization_as_block_threshold = 128;
                    dim3 ScoringPathThreads(cssize.width, 1, 512/cssize.width);
                    if(cssize.width>path_optimization_as_block_threshold){
                        ScoringPathThreads = dim3(32,1,16);
                    }

                    dim3 ScoringPathBlocks(
                        max(1,(cssize.width+ScoringPathThreads.x-1)/ScoringPathThreads.x),
                        1,
                        max(1,(cssize.samples+ScoringPathThreads.z-1)/ScoringPathThreads.z));

                    // evaluate the best fit in local window
                    dim3 OptimThreads(16,1,16);
                    dim3 OptimBlocks(1,1,max(1,(cssize.samples+OptimThreads.z-1)/OptimThreads.z));
                    
                    dim3 GetPosThreads(min(32,opt.interpolations),max(1, (32+opt.interpolations-1)/opt.interpolations),16);
                    // calculate the mean segment size as a heuristic for optimal y-blockdim
                    // we can safely assume that each segment is at least of size 1, therefore we will always launch at least one block
                    size_t meanSegmentSize = 0;
                    for(int i = 0; i< segments[chnl].size()-1;++i){
                        meanSegmentSize += segments[chnl][i+1]-segments[chnl][i];
                    }
                    meanSegmentSize = (meanSegmentSize+segments[chnl].size()-1)/segments[chnl].size();

                    dim3 GetPosBlocks(
		                max(1,(opt.interpolations+GetPosThreads.x-1)/GetPosThreads.x),
			            max(1ul,(meanSegmentSize+GetPosThreads.y-1)/GetPosThreads.y),
			            max(1ul,(num_fronts+GetPosThreads.z-1)/GetPosThreads.z));
                    if(opt.verbose_timer_calc){
                        copy_timer1.reset();
                        copy_timer1.start();
                    }
                    thrust::copy(rmm::exec_policy_nosync(calc_stream), front_coords[chnl].begin(), front_coords[chnl].end(), coord_d.begin());
                    thrust::copy(rmm::exec_policy_nosync(calc_stream), offsets[chnl].begin(), offsets[chnl].end(), offsets_d.begin());
                    thrust::copy(rmm::exec_policy_nosync(calc_stream), segments[chnl].begin(), segments[chnl].end(), segments_d.begin());
                    
                    if(opt.verbose_timer_calc){
                        copy_timer1.print();
                    }

                    if(chnl == opt.out_chnls.size()-1)
                        cudaEventRecord(data_processed, calc_stream);
                    CUERR;
                    if(opt.verbose_timer_calc){
                        thrust_prefill_timer.reset();
                        thrust_prefill_timer.start();
                    }

                    thrust::sequence(rmm::exec_policy_nosync(calc_stream), positions_d.begin(), positions_d.begin()+num_fronts,0);
                    
                    thrust::fill(rmm::exec_policy_nosync(calc_stream),diffs.begin(),diffs.end(), 1000000);
                    if(opt.verbose_timer_calc){
                        thrust_prefill_timer.print();
                    }
                    
                    if(opt.verbose_timer_calc){
                        get_pos_and_dir_timer.reset();
                        get_pos_and_dir_timer.start();
                    }
                    getPosAndDirWKey<<<GetPosBlocks, GetPosThreads, 0, calc_stream>>>(coord_d.data(), offsets_d.data(), segments_d.data(), bases.data(), dirs.data(), keys_d.data(), positions_d.data(),num_fronts, opt.rotations, opt.smoothing_steps, opt.interpolations);
                    CUERR;
                    // Get The Cross section data for each basis and direction
                    cudaStreamWaitEvent(calc_stream, texture_copied);
                    if(opt.verbose_timer_calc){
                        get_pos_and_dir_timer.print();
                    }
                    
                    if(opt.verbose_timer_calc){
                        cross_section_timer.reset();
                        cross_section_timer.start();
                    }
                    getCSValueFromTexWOrient<<<CSBlocks,CSThreads, 0, calc_stream>>>(grid_vec, grid_loc_vec, bases.data(), dirs.data(), needs_flip.data(), orientations_d.data(), myTexs[0] ,myTexs[1], myTexs[2], opt.reversal[chnl], opt.step_km, opt.dir_step_km);
                    flipDir<<<max(1,(cssize.samples+maxThreadsInABlock-1)/maxThreadsInABlock), maxThreadsInABlock, 0, calc_stream>>>(dirs.data(), needs_flip.data(), cssize.samples);
                    if(opt.verbose_timer_calc){
                        cross_section_timer.print();
                    }
                    cudaEventRecord(texture_read, calc_stream);
                    CUERR;
                    if(opt.adjust_orientation){
                        if(opt.verbose_timer_calc){
                            adjust_orientation_timer.reset();
                            adjust_orientation_timer.start();
                        }
                        // adjust orientations
                        // get mean orientation of each key
                        thrust::reduce_by_key(rmm::exec_policy(calc_stream), keys_d.begin(), keys_d.begin()+num_smp, orientations_d.begin(), tgt_keys_d.begin(), mean_orientations_d.begin());
                        // readjust mean orientation of each key
                        adjustOrientation<<<CSBlocks,CSThreads, 0, calc_stream>>>(grid_vec, grid_loc_vec, keys_d.data(), orientations_d.data(), tgt_keys_d.data(), mean_orientations_d.data(), segments_d.data(), dirs.data(), opt.interpolations, opt.smoothing_steps*opt.rotations);

                        CUERR;
                        if(opt.verbose_timer_calc){
                            adjust_orientation_timer.print();
                        }
                    }

                    // Write Cross Sections
                    if(opt.baroclinity_eval || opt.write_cross_sections){
                        auto h_vec = fronts::pinned_host_vector<typename decltype(vec)::value_type>(cssize.width*cssize.samples*cssize.height);
                        cudaMemcpy(thrust::raw_pointer_cast(h_vec.data()), vec.data(), cssize.width*cssize.samples*cssize.height*sizeof(typename decltype(vec)::value_type), cudaMemcpyDeviceToHost);
                        std::string infilename = background_files[file_id].filename().string().substr(2,11);
                        reader.write(output_fold / fs::path(std::string(infilename+"_"+opt.out_chnls[chnl]+"_cs.bin")), thrust::raw_pointer_cast(h_vec.data()), h_vec.size());
                    }
                    
                    // fill the scoring matrix with values
                    CUERR;
                    if(opt.verbose_timer_calc){
                        calc_score_timer.reset();
                        calc_score_timer.start();
                    }
                    calculateScore<<<ScoringBlocks, ScoringThreads, 0, calc_stream>>>(grid_vec, grid_scores, opt.filter_invalid_samples, opt.mean_width); 
                    if(opt.verbose_timer_calc){
                        calc_score_timer.print();
                    }
                    if((!opt.verbose_timer) && opt.write_scores){
                        if(opt.verbose_timer_calc){
                            eval_score_timer.reset();
                            eval_score_timer.start();
                        }
                        // for comparison reasons, I want to evaluate against a 10 pixel pre and post front temperature
                        // if the scores are not to be written, we can skip this
                        calculateScore<<<ScoringBlocks, ScoringThreads, 0, calc_stream>>>(grid_vec, grid_out_scores, opt.filter_invalid_samples, 10); 
                        if(opt.verbose_timer_calc){
                            eval_score_timer.print();
                        }
                    }

                    CUERR;
                    
                    // calculate the opt path scoring for each pixel
                    if(opt.optimize_paths){
                        if(opt.verbose_timer_calc){
                            optimal_path_timer.reset();
                            optimal_path_timer.start();
                        }
                        if(cssize.width > path_optimization_as_block_threshold){
                        // cssize.width threads are not in a single block. We need inter kernel synchronization (kernel launches)
                            for(int l = 1; l<cssize.height; ++l){
                                calculateScorePath<<<ScoringPathBlocks, ScoringPathThreads, 0, calc_stream>>>(grid_scores, l); 
                            }
                        }
                        else{
                            // cssize.width threads fit in a single block. Use intra kernel synchronization (syncthreads)
                            calculateScorePathBlock<<<ScoringPathBlocks, ScoringPathThreads, ScoringPathThreads.x*ScoringPathThreads.z*sizeof(DTYPE_), calc_stream>>>(grid_scores);
                        }
                    
                        CUERR;
                        if(opt.verbose_timer_calc){
                            optimal_path_timer.print();
                        }
                    }

                    cudaStreamWaitEvent(calc_stream, local_results_copied);
                    CUERR;
                    if(opt.verbose_timer_calc){
                        optimize_timer.reset();
                        optimize_timer.start();
                    }
                    // evaluate the best fit in local window
                    getMinScoresPerHeight<<<OptimBlocks, OptimThreads, 0, calc_stream>>>(grid_scores, grid_out_scores, grid_septs, grid_diffs, grid_loc_vec, grid_seplocs, opt.reversal[chnl], true, opt.static_eval, opt.random_eval, opt.optimization_window_size);
                    CUERR;
                    if(opt.verbose_timer_calc){
                        optimize_timer.print();
                    }
                    
                    
                    cudaEventRecord(local_results_calculated, calc_stream);
                    cudaStreamWaitEvent(copyD2H_stream, local_results_calculated);
                    if(opt.verbose_timer_calc){
                        calc_wait_timer_write.reset();
                        calc_wait_timer_write.start();
                    }

                    #ifndef USE_ATOMICS
                    {
                        std::unique_lock overwrite_lk(overwrite_m);
                        overwrite_cv.wait(overwrite_lk, [&]{return can_overwrite_result;});
                        can_overwrite_result = false;
                    }
                    #else
                    while(myWriteState){
                        can_overwrite_result.wait(true);
                        myWriteState = can_overwrite_result.test_and_set();
                    }
                    #endif
                    if(opt.verbose_timer_calc){
                        calc_wait_timer_write.print();
                    }
                    
                    saveSize = cssize.samples*cssize.height;
                    sampleCount = cssize.samples;
                    // write directions of each sample point => This way we can later extract the horizontal normal direction at each sample
                    if(opt.verbose_timer_calc){
                        copy_timer2.reset();
                        copy_timer2.start();
                    }

                    if(opt.write_seps) thrust::copy(rmm::exec_policy_nosync(copyD2H_stream), septs.begin(), septs.begin()+saveSize, seps_h.begin());
                    if(opt.write_scores) thrust::copy(rmm::exec_policy_nosync(copyD2H_stream), diffs.begin(), diffs.begin()+saveSize, diffs_h.begin());
                    if(opt.write_coords || opt.write_netCDF) thrust::copy(rmm::exec_policy_nosync(copyD2H_stream), seplocs.begin(), seplocs.begin()+saveSize, locs_h.begin());
                    if(opt.write_dirs || opt.write_netCDF) thrust::copy(rmm::exec_policy_nosync(copyD2H_stream), dirs.begin(), dirs.begin()+cssize.samples, dirs_h.begin());
                    
                    if(opt.verbose_timer_calc){
                        copy_timer2.print();
                    }

                    
                    cudaEventRecord(local_results_copied, copyD2H_stream);
                    
                    #ifndef USE_ATOMICS
                    {
                        // inform the worker that some data exists
                        std::unique_lock lk(write_m);
                        can_write = true;
                        write_cv.notify_one();
                    }
                    #else
                    can_write.clear();
                    can_write.notify_one();
                    #endif

                } // end chnl
                #ifndef USE_ATOMICS
                {
                    std::unique_lock overwrite_lk(overwrite_m);
                    overwrite_cv.wait(overwrite_lk, [&]{return can_overwrite_result;});
                }
                #else
                while(myWriteState){
                    can_overwrite_result.wait(true);
                    myWriteState = can_overwrite_result.test_and_set();
                }
                #endif
                if(opt.verbose_timer_calc){
                    calc_wait_timer_write.print();
                }
                cudaEventRecord(results_written_event, calc_stream);
                if(opt.verbose_timer_calc) process_file_timer.print();
            }
        } // end file
        process_all_timer.print();
    
    }); // working end
    // asynchronous writing of 3D frontal coordinates
    auto res3 = std::async(std::launch::async, [&](){
        cudaSetDevice(opt.gpuID);
        #ifdef USE_ATOMICS
            bool myState = true;
        #endif
        write_all_timer.reset();
        write_all_timer.start();
        for(int file_id = 0;file_id< background_files.size(); file_id++){
            if(opt.verbose_timer_write) write_file_timer.reset();
            if(opt.verbose_timer_write) write_file_timer.start();
            std::string infilename = background_files[file_id].filename().string().substr(2,11);
            for(int chnl=0; chnl < opt.out_chnls.size(); ++chnl){
                if(opt.verbose_timer_write){
                    write_wait_timer_calc.reset();
                    write_wait_timer_calc.start();
                }
                #ifndef USE_ATOMICS
                {
                    std::unique_lock write_lk(write_m);
                    write_cv.wait(write_lk, [&]{return can_write;});
                    can_write = false;
                }
                #else
                    while(myState){
                        can_write.wait(true);
                        myState = can_write.test_and_set();
                    }
                #endif
                cudaEventSynchronize(local_results_copied); CUERR;
                if(opt.verbose_timer_write){
                    write_wait_timer_calc.print();
                }
                if(opt.verbose_timer_write){
                    write_other_timer.reset();
                    write_other_timer.start();
	        	}
                
                if(opt.write_seps) reader.write(output_fold / fs::path(std::string(infilename+"_"+opt.out_chnls[chnl]+"_seps.bin")), thrust::raw_pointer_cast(seps_h.data()), saveSize);
                if(opt.write_scores) reader.write(output_fold / fs::path(std::string(infilename+"_"+opt.out_chnls[chnl]+"_diffs.bin")), thrust::raw_pointer_cast(diffs_h.data()), saveSize);
                if(opt.write_coords) reader.write(output_fold / fs::path(std::string(infilename+"_"+opt.out_chnls[chnl]+".bin")), thrust::raw_pointer_cast(locs_h.data()),saveSize);
                if(opt.write_dirs) reader.write(output_fold / fs::path(std::string(infilename+"_"+opt.out_chnls[chnl]+"_dirs.bin")), thrust::raw_pointer_cast(dirs_h.data()), sampleCount);
                if(opt.verbose_timer_write){
		            write_other_timer.print();
		        }

                if(opt.write_netCDF){
                    if(opt.verbose_timer_write){
                        create_netCDF_grid_timer.reset();
                        create_netCDF_grid_timer.start();
                    }
                        
                    convertPositionsToGridPoints(locs_h, dirs_h, output_data_buffer[0], packingParameter[0], opt.out_chnls[chnl], sampleCount, opt);
                    if(opt.verbose_timer_write) create_netCDF_grid_timer.print();
                }
                #ifndef USE_ATOMICS
                {
                    // inform the worker that some data exists
                    std::unique_lock lk(overwrite_m);
                    can_overwrite_result = true;
                    overwrite_cv.notify_one();
                }
                #else
                    can_overwrite_result.clear();
                    can_overwrite_result.notify_one();
                #endif
		
            }
            
            if(opt.write_netCDF){ 
                if(opt.verbose_timer_write){
                    write_wait_timer_netCDF.reset();
                    write_wait_timer_netCDF.start();
		        }
                #ifndef USE_ATOMICS
                {
                    // restrict netCDF access (threadsafety!)
                    std::unique_lock writeNc_lk(netCDFIO_m);
                    netCDFIO_cv.wait(writeNc_lk, [&]{return can_access_netCDF;});
                    can_access_netCDF = false;
                }
                #else
                    while(myState){
                        can_access_netCDF.wait(true);
                        myState = can_access_netCDF.test_and_set();
                    }
                #endif
                if(opt.verbose_timer_write){
		            write_wait_timer_netCDF.print();
		        }
                if(opt.verbose_timer_write){
                    write_netCDF_timer.reset();
                    write_netCDF_timer.start();
		        }
                ncWriter.write(output_fold / fs::path(std::string(infilename+".nc")), output_data_buffer[0], gridDims, packingParameter[0]);
                if(opt.verbose_timer_write){
		            write_netCDF_timer.print();
		        }
                #ifndef USE_ATOMICS
                {
                    // release netCDF access (threadsafety!)
                    std::unique_lock writeNc_lk(netCDFIO_m);
                    can_access_netCDF = true;
                    netCDFIO_cv.notify_one();
                }
                #else
                    can_access_netCDF.clear();
                    can_access_netCDF.notify_one();
                #endif
                if(opt.verbose_timer_write){
		            write_clear_grid_timer.reset();
                    write_clear_grid_timer.start();
		        }
                for(int chnl=0; chnl < opt.out_chnls.size(); ++chnl){
                    clearPositionsToGridPoints(locs_h, dirs_h, output_data_buffer[0], packingParameter[0], opt.out_chnls[chnl], sampleCount, opt);
                }
                if(opt.verbose_timer_write) write_clear_grid_timer.print();
            }
            if(opt.verbose_timer_write) write_file_timer.print();
        }
        write_all_timer.print();
    });

    // Join
    for(auto& a: reader_threads)
        a.get();
    res2.get();
    res3.get();
    overall_timer.print();
    calc_log.close();
    read_log.close();
    write_log.close();
}

