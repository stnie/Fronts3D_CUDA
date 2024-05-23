#pragma once

#include "helper_3d.cuh"
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>
#include "rmm/device_uvector.hpp"


// container for vector storage on device and host
namespace fronts{
    using pinned_memory_resource = thrust::cuda::universal_host_pinned_memory_resource;
    template<class T>
    using pinned_allocator = thrust::mr::stateless_resource_allocator<T, pinned_memory_resource>;
    template<class T>
    using pinned_host_vector = thrust::host_vector<T, pinned_allocator<T>>;
    template<class T>
    using device_vector = rmm::device_uvector<T>;
}
