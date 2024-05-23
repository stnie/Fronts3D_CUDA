#pragma once
#include <cuda_helpers.cuh>
namespace helpers{

    DEVICEQUALIFIER INLINEQUALIFIER
    dim3 global_thread_3D_pos() noexcept
    {
        return
            dim3(blockDim.x*blockIdx.x+threadIdx.x, 
                    blockDim.y*blockIdx.y+threadIdx.y, 
                    blockDim.z*blockIdx.z+threadIdx.z);
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    dim3 global_thread_3D_pos_z_fastest() noexcept
    {
        return
            dim3(blockDim.z*blockIdx.z+threadIdx.z, 
                    blockDim.y*blockIdx.y+threadIdx.y, 
                    blockDim.x*blockIdx.x+threadIdx.x);
    }
    
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_in(dim3 pos, dim3 upperLimit){
        return pos.x<upperLimit.x && pos.y<upperLimit.y && pos.z<upperLimit.z;
    }
    
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_in(T pos, dim3 lowerLimit, dim3 upperLimit){
        return pos.x<upperLimit.x && pos.y<upperLimit.y && pos.z<upperLimit.z &&
            pos.x>=lowerLimit.x && pos.y>=lowerLimit.y && pos.z>=lowerLimit.z;
    }
    };