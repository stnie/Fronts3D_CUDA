#pragma once
#include <cuda_helpers.cuh>
#include <cufronts/types/math_helper.cuh>
#include "cufronts/processing/util/processingConfig.cuh"
#include "cufronts/processing/util/helper_3d.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <curand_kernel.h>

template<class T>
DEVICEQUALIFIER
T optimizeSingleLevel(fronts::crossSectionGrid<T> scores, fronts::crossSectionGrid<T> out_scores, fronts::outputGrid<int> outs, fronts::outputGrid<T> minScores, fronts::crossSectionGrid<val2<T>> locations, fronts::outputGrid<val2<T>> sep_locations, int* minPosses, int leveloffset, int leftOff, int window_size){
    auto myPos = helpers::global_thread_3D_pos_z_fastest();
    dim3 dims (scores.extent(0),scores.extent(1),scores.extent(2));
    auto width = dims.z;
    auto sampleIndex = myPos.x;
    
    const cg::thread_block_tile<16> myGroup = cg::tiled_partition<16>(cg::this_thread_block());
    int w_step_size = window_size;
    int num_w_steps = (window_size+myGroup.size()-1)/myGroup.size();
    int minPos = 0;
    T myMin = INVALID_SCORE;

    for(int w_step = 0; w_step < num_w_steps; ++w_step){
        int widthPos = leftOff+myGroup.thread_rank()+w_step*w_step_size;
        T myVal = widthPos < width ? scores(sampleIndex, leveloffset, widthPos) : INVALID_SCORE;
        
        myVal = myGroup.thread_rank() < window_size ? myVal : INVALID_SCORE;
        T myIterMin = cg::reduce(myGroup, myVal, [](T x, T y){return min(x,y);});
        if(myIterMin < myMin){
            myMin = myIterMin;
            auto minMask = myGroup.ballot(myMin == myVal);
            minPos = __ffs(minMask)+w_step*w_step_size;
        }
    }
    
    // each leading thread per group writes the groups minimum into shared
    if(threadIdx.x == 0)
        minPosses[threadIdx.z] = myMin == INVALID_SCORE ? 0 : minPos;
    __syncthreads();
    int groupMinPos = minPosses[myGroup.thread_rank()];
    __syncthreads();
    int positivePos = groupMinPos>0;

    int cnts = cg::reduce(myGroup, positivePos, [](int x, int y){return x+y;});
    // __ffs outputs the position 1 based. We need to correct this offset.
    int meanPos = cg::reduce(myGroup, groupMinPos, [](int x, int y){return x+y;})/cnts-1;
    minPos -= 1;
    
    // if cnts is zero (i.e. no valid locations within the group)
    // set the mean Pos to window_size/2 (results in no movement)
    if(cnts == 0) meanPos = window_size/2;

    // only the first thread of a group needs to write to global
    if(threadIdx.x == 0){
        bool is_invalid = myMin == INVALID_SCORE;
        if(is_invalid){
            outs(leveloffset, sampleIndex) = INVALID_OFFSET;
            minScores(leveloffset, sampleIndex) = myMin;
            sep_locations(leveloffset, sampleIndex) = INVALID_COORDS;
        }
        else{
            outs(leveloffset, sampleIndex) = minPos+leftOff;
            minScores(leveloffset, sampleIndex) = out_scores(sampleIndex, leveloffset, minPos+leftOff);
            sep_locations(leveloffset, sampleIndex) = locations(sampleIndex, 0, minPos+leftOff);
        }
    }
    return meanPos;
}



template<class T>
GLOBALQUALIFIER
void getMinScoresPerHeight(fronts::crossSectionGrid<T> scores, fronts::crossSectionGrid<T> out_scores, fronts::outputGrid<int> outs, fronts::outputGrid<T> minScores, fronts::crossSectionGrid<val2<T>> locations, fronts::outputGrid<val2<T>> sep_locations, bool reverse, bool bottomToTop = true, bool static_eval = false, bool random_eval = false,  int default_window_size = 16){
    auto myPos = helpers::global_thread_3D_pos_z_fastest();
    dim3 dims (scores.extent(0),scores.extent(1),scores.extent(2));
    
    curandState state;
    int totalPos = myPos.x*blockDim.x*gridDim.x*blockDim.y*gridDim.y+myPos.y*blockDim.x*gridDim.x+myPos.z;
    // fixed seed for reproducibility
    curand_init(42, totalPos, 0, &state);
    
    // x and z dim currently need to be set to 16!
    __shared__ int minPosses[16];
    if(threadIdx.z == 0){
        minPosses[threadIdx.x] = 0;
    }
    __syncthreads();

    int window_size = default_window_size;
    if(random_eval || static_eval)
        window_size = 1;
    // this is correct as long as blockDim.x is 32*k
    if(helpers::is_in(myPos, dim3(0,0,0), dims)){
        auto height = dims.y;
        auto width = dims.z;
        
        int leftOff = width/2-window_size/2;
        if(random_eval){
            int myOffset = curand_uniform(&state)*dims.z;
            leftOff = myOffset;
            leftOff = max(0,min(leftOff, width-window_size));
        }
        
        for(int h = 0; h < height; ++h){
            int leveloffset = height-h-1;
            bool optimizeDownwards = false;
            if(optimizeDownwards){
                leveloffset = h;
            }

            T meanPos = optimizeSingleLevel(scores, out_scores, outs, minScores, locations, sep_locations, minPosses, leveloffset, leftOff, window_size);
            
            if(random_eval && static_eval){
                int myOffset = curand_uniform(&state)*default_window_size;
                leftOff += myOffset-default_window_size/2; 
            }
            else if(random_eval){
                int myOffset = curand_uniform(&state)*dims.z;
                leftOff = myOffset;
            }
            else{
                int window_shift = meanPos-window_size/2;
                leftOff += window_shift;
            }
            leftOff = max(0,min(leftOff, width-window_size));
        }
    }
}
