#pragma once
#include "rmm/device_uvector.hpp"
#include <iostream>


template<class T>
cudaTextureObject_t createTex(dim3 dims){

    auto width = dims.z;
    auto height = dims.y;
    auto depth = dims.x;

    cudaChannelFormatDesc channelDesc = 
        cudaCreateChannelDesc<T>();
    cudaArray_t dstArray;

    cudaExtent extent = make_cudaExtent(width, height, depth);
    cudaMalloc3DArray(&dstArray, &channelDesc, extent, cudaArrayLayered);
    
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dstArray;

    // Specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

template<class T>
void updateTex(T* data, cudaTextureObject_t& tex, dim3 dims, cudaStream_t& stream, int offset = 0){

    auto width = dims.z;
    auto height = dims.y;
    auto depth = dims.x;

    cudaResourceDesc resDesc;
    cudaGetTextureObjectResourceDesc(&resDesc, tex);
    cudaArray_t dstArray = resDesc.res.array.array;

    cudaExtent extent = make_cudaExtent(width, height, depth);

    cudaMemcpy3DParms memcpy3dParms = {0};
    memcpy3dParms.srcPtr = make_cudaPitchedPtr(data+offset, width*sizeof(T), width, height);
    memcpy3dParms.dstArray = dstArray;
    memcpy3dParms.extent = extent;
    memcpy3dParms.kind = cudaMemcpyDeviceToDevice;

    int activeDevice;
    cudaGetDevice(&activeDevice);

    cudaMemcpy3DAsync(&memcpy3dParms, stream); CUERR;
}


template<class T>
std::vector<cudaTextureObject_t> createTextures(dim3 dataDim){
    // setup background texture (theta_e)
    auto tex = createTex<T>(dataDim);
    // setup lat lon texture
    auto location_tex = createTex<T>(dim3(2,dataDim.y,dataDim.z));
    // setup direction estimation background texture 
    auto dir_base_tex = createTex<T>(dim3(1,dataDim.y,dataDim.z));
    
    // clear temporary vectors
    return std::vector<cudaTextureObject_t>{tex, location_tex, dir_base_tex};
}

template<class vec_t>
void updateTextures(vec_t& data, vec_t& coord_data, std::vector<cudaTextureObject_t>& texs, dim3 dataDim, cudaStream_t& stream){
    // setup background texture (theta_e)
    updateTex(data.data(), texs[0], dataDim, stream);

    // setup lat lon texture
    updateTex(coord_data.data(), texs[1], dim3(2,dataDim.y,dataDim.z), stream);

    // setup direction estimation background texture 
    int dir_base_off = dataDim.y*dataDim.z*(8);
    updateTex(data.data(), texs[2], dim3(1,dataDim.y,dataDim.z), stream, dir_base_off);

}

