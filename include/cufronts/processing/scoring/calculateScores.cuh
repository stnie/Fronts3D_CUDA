
#pragma once
#include <cuda_helpers.cuh>
#include <cufronts/types/math_helper.cuh>
#include "cufronts/processing/util/processingConfig.cuh"
#include "cufronts/processing/util/helper_3d.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda/std/mdspan>



namespace cg = cooperative_groups;

template<class T>
HOSTDEVICEQUALIFIER
T getGradient(T* data_in, dim3 dim, dim3 pos, dim3 diff){
    // x fastest, y mid, z slowest
    T x = data_in[(pos.x-diff.x) + (pos.y-diff.y)*dim.x + (pos.z-diff.z)*dim.x*dim.y];
    T y = data_in[(pos.x+diff.x) + (pos.y+diff.y)*dim.x + (pos.z+diff.z)*dim.x*dim.y];
    return (y-x)/2;
}


template<class vec_t>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
typename vec_t::value_type getGradient(const vec_t& data_in, const dim3& dim, const dim3& pos, const dim3& diff){
    const typename vec_t::value_type x = data_in((pos.x-diff.x), (pos.y-diff.y), (pos.z-diff.z));
    const typename vec_t::value_type y = data_in((pos.x+diff.x), (pos.y+diff.y), (pos.z+diff.z));
    return (y-x)/2;
}

template<class vec_t>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
typename vec_t::value_type getVariances(const vec_t& data_in, const dim3& dim, const dim3& pos, const int dist = 10){
    typename vec_t::value_type leftSum = 0;
    typename vec_t::value_type rightSum = 0;
    typename vec_t::value_type leftSqSum = 0;
    typename vec_t::value_type rightSqSum = 0;
    int leftCnt = 0;
    int rightCnt = 0;
    for (int z = 1; z < dist+1; z++){
        if ((int)pos.z -z >= 0){
            auto lval = data_in(pos.x,pos.y, pos.z-z);
            leftSum += lval;
            leftSqSum += lval*lval;
            leftCnt += 1;
        }
        if ((int)pos.z + z < dim.z){
            auto rval = data_in(pos.x,pos.y, pos.z+z);
            rightSum += rval;
            rightSqSum += rval*rval;
            rightCnt += 1;
        }
    }
    if(leftCnt>0){
        leftSqSum /= leftCnt;
        leftSum /= leftCnt;
    }
    if(rightCnt>0){
        rightSqSum /= rightCnt;
        rightSum /= rightCnt;
    }
    auto leftVar = leftSqSum - leftSum*leftSum;
    auto rightVar = rightSqSum - rightSum*rightSum;
    return sqrt(rightVar*rightVar+leftVar*leftVar);
}

template<class vec_t>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
typename vec_t::value_type getDiffs(const vec_t data_in, const dim3& dim, const dim3& pos, const int diff = 10){
    typename vec_t::value_type leftSum = 0;
    typename vec_t::value_type rightSum = 0;

    int leftCnt = 0;
    int rightCnt = 0;
    for (int z = 1; z < diff+1; z++){
        if ((int)pos.z - z >= 0){
            auto lval = data_in(pos.x,pos.y, pos.z-z);
            leftSum += lval;
            leftCnt += 1;
        }
        else{
            auto lval = data_in(pos.x,pos.y, 0);
            leftSum += lval;
            leftCnt += 1;
        }
        if ((int)pos.z +z < dim.z){
            auto rval = data_in(pos.x, pos.y, pos.z+z);
            rightSum += rval;
            rightCnt += 1;
        }
        else{
            auto rval = data_in(pos.x, pos.y, dim.z-1);
            rightSum += rval;
            rightCnt += 1;
        }
    }
    if(leftCnt>0)
        leftSum /= leftCnt;
    if(rightCnt>0)
        rightSum /= rightCnt;
    
    return rightSum-leftSum;
}

template<class T>
GLOBALQUALIFIER
void calculateScore(fronts::crossSectionGrid<T> cs_data, fronts::crossSectionGrid<T> scores, bool filter_invalid_samples, int mean_width = 5){
    // swap x and z, such that the fastest changing thread idx 
    // also has the fastest changing position idx (as used by mdspan)
    auto myPos = helpers::global_thread_3D_pos_z_fastest();
    
    float w1 = 0;
    float w2 = 0;
    float w3 = 1;

    dim3 dims(cs_data.extent(0),cs_data.extent(1),cs_data.extent(2));
    auto samples = dims.x;
    auto height = dims.y;
    auto width = dims.z;

    // dims is: samples, height, width
    T myScore = INVALID_SCORE;
    if(helpers::is_in(myPos, dim3(0,0,0), dims)){
        if(helpers::is_in(myPos, dim3(0,0,1), dim3(samples, height, width-1))){
            if(cs_data(myPos.x,myPos.y,myPos.z) != INVALID_DATA){
                auto grad = getGradient(cs_data, dims, myPos, dim3(0,0,1));
                // currently not used as w2 is set to 0 
                auto vars = getVariances(cs_data, dims, myPos, 20);
                auto diffs = getDiffs(cs_data, dims, myPos, mean_width);
                
                if(filter_invalid_samples){
                    auto myThresh = 0;
                    auto myThresh2 = 0;
                    myScore = ((grad < myThresh) && (diffs<myThresh2)) ? grad*w1+vars*w2+diffs*w3 : INVALID_DYN_SCORE;
                }
                else{
                    myScore = grad*w1+vars*w2+diffs*w3;
                }
            }
        }
        scores(myPos.x,myPos.y,myPos.z)  = myScore;        
    }
}


template<class T>
GLOBALQUALIFIER
void calculateScorePath(fronts::crossSectionGrid<T> scores, int level, int leveldir = -1){
    // optimize from highest to lowest level
    // at the end we obtain the best path scoring for any bottom level
    // this allows an easy optimization from bottom up
    auto myPos = helpers::global_thread_3D_pos_z_fastest();
    dim3 dims(scores.extent(0),scores.extent(1),scores.extent(2));


    constexpr const  int searchrange = 8;
    const int leveloffset = level;
    
    constexpr const int samples_per_block = 16; //blockDim.z;
    const int width_per_block = 32 ; // blockDim.x;
    constexpr const int width_plus_searchrange = width_per_block+2*searchrange;
    myPos.y = leveloffset;
    __shared__ T previousRow[width_plus_searchrange*samples_per_block];
    int offset = 0;
    if(myPos.x < dims.x){
        while(threadIdx.x+width_per_block*offset < width_plus_searchrange){
            if(myPos.z+width_per_block*offset >= searchrange)
	            previousRow[threadIdx.x+width_per_block*offset+threadIdx.z*width_plus_searchrange] = scores(myPos.x, myPos.y+leveldir, myPos.z-searchrange+width_per_block*offset);
	        offset+=1;
	    }
    }
    __syncthreads();
    if(helpers::is_in(myPos, dim3(0,0,0), dims)){
        auto myScore = scores(myPos.x, myPos.y, myPos.z);
        auto myOptScore = myScore+leveloffset*INVALID_DYN_SCORE;
        for(int subpos = -searchrange; subpos < searchrange+1; ++subpos){
            int myPrevZPos = myPos.z+subpos;
            if(myPrevZPos >= 0 && myPrevZPos < dims.z){
                auto testScore = previousRow[threadIdx.x+searchrange+subpos+threadIdx.z*width_plus_searchrange];
                T offset_penalty = abs((T)subpos*subpos);
                myOptScore = min(myOptScore, myScore+testScore+offset_penalty);
            }
        }
        scores(myPos.x, myPos.y, myPos.z) = myOptScore;
    }
}

template<class T>
GLOBALQUALIFIER
void calculateScorePathBlock(fronts::crossSectionGrid<T> scores, int leveldir = -1){
    // optimize from highest to lowest level
    // at the end we obtain the best path scoring for any bottom level
    // this allows an easy optimization from bottom up
    // this kernel assumes that a whole row of a cross section can fit in a block. => We can use syncthreads instead of launching a kernel per level
    auto myPos = helpers::global_thread_3D_pos_z_fastest();
    auto tmp = myPos.z; myPos.z = myPos.x; myPos.x = tmp;
    dim3 dims(scores.extent(0),scores.extent(1),scores.extent(2));
    int searchrange = 8;
    extern __shared__ T previousRow[];
    for(int level = 1; level<dims.y; ++level){
	myPos.y = level;
	if(myPos.x < dims.x){
	    previousRow[threadIdx.x+threadIdx.z*dims.z] = scores(myPos.x, myPos.y, myPos.z);
	}
	__syncthreads();
        if(helpers::is_in(myPos, dim3(0,0,0), dims)){
            auto myScore = scores(myPos.x, myPos.y, myPos.z);
            auto myOptScore = myScore+level*INVALID_DYN_SCORE;
            for(int subpos = -searchrange; subpos < searchrange+1; ++subpos){
                int myPrevZPos = myPos.z+subpos;
                if(myPrevZPos >= 0 && myPrevZPos < dims.z){
                    auto testScore = previousRow[threadIdx.x+threadIdx.z*dims.z];
                    T offset_penalty = abs((T)subpos*subpos);
                    myOptScore = min(myOptScore, myScore+testScore+offset_penalty);
                }
            }
            scores(myPos.x, myPos.y, myPos.z) = myOptScore;
	}
	__syncthreads();
    }
}