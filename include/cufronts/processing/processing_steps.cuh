#pragma once
#include <vector>
#include <mutex>
#include <atomic>
#include <future>
#include <iostream>

#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/adjacent_difference.h>

#include <timers.cuh>
#include <cufronts/types/typedefs.hpp>
#include <cufronts/IO/BinaryIO.hpp>


#include "rmm/device_uvector.hpp"
#include "rmm/device_vector.hpp"
#include "rmm/mr/device/cuda_async_memory_resource.hpp"
#include "rmm/exec_policy.hpp"

#include "cufronts/processing/util/helper_3d.cuh"
#include "cufronts/processing/util/container_typedefs.hpp"
#include "cufronts/processing/util/extent.hpp"
#include "cufronts/processing/util/options.hpp"
#include <cufronts/processing/util/conversion_helper.cuh>

void getDiffs(frontIO::frontMap<DTYPE_>& fronts_container, std::vector<std::string> in_chnls, 
    std::vector<fronts::pinned_host_vector<val2<DTYPE_>>>& front_coords, 
    std::vector<fronts::pinned_host_vector<size_t>>& offsets,
    std::vector<fronts::pinned_host_vector<size_t>>& segments,
    std::vector<fronts::pinned_host_vector<size_t>>& sortOffsets,
    int level, int factor);

void newMethodPrealloc(std::vector<fs::path>& background_files, std::vector<fs::path>& front_files, fs::path output_fold, fronts::options opt);
