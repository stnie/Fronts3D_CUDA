#pragma once
#include <cufronts/types/typedefs.hpp>
#include <random>

struct CoordinateNoise
{
    std::mt19937 generator;
    void initialize(){
		// fixed seed for reproducibility
        std::seed_seq seed{ 42,  43,  44,  45,  46,  47,  48,  49};
        generator = std::mt19937(seed);
    }


    void addNoiseToCoords(std::vector<fronts::pinned_host_vector<val2<DTYPE_>>>& front_coords, std::vector<fronts::pinned_host_vector<size_t>>& offsets, float stddev, bool perVertex = false){
        std::normal_distribution<> normal_dist(0, stddev);
        if(perVertex){
            for(auto& front_coord: front_coords){
                for(auto& coord_pair: front_coord){
    	        float x = normal_dist(generator);
    	        float y = normal_dist(generator);
                    coord_pair.x += x;
                    coord_pair.y += y;
                }
    	    }
        }
        else{
			int chnl = 0;	
			float x = 0;
			float y = 0;
			std::vector<val2<float>> occOffs;
            for(auto& front_coord: front_coords){
				if(chnl == 2){
					int offset = 0;
					int segmentIdx = 0;
					for(auto& coord_pair: front_coord){
						if(offset == offsets[chnl][segmentIdx]){
							x = normal_dist(generator);
							y = normal_dist(generator);
							occOffs.push_back(val2<float>(x,y));
							segmentIdx+=1;
						}
						coord_pair.x += x;
						coord_pair.y += y;
						offset += 1;
					}
				}
				else if(chnl == 3){
					int offset = 0;
					int segmentIdx = 0;
        	        for(auto& coord_pair: front_coord){
		    			if(offset == offsets[chnl][segmentIdx]){
    	                	x = occOffs[segmentIdx].x;
    	                	y = occOffs[segmentIdx].y;
		        			segmentIdx+=1;
		    			}
						coord_pair.x += x;
						coord_pair.y += y;
						offset += 1;
					}
				}
				else{	
					int offset = 0;
					int segmentIdx = 0;
        	        for(auto& coord_pair: front_coord){
		    			if(offset == offsets[chnl][segmentIdx]){
    	                	x = normal_dist(generator);
    	                	y = normal_dist(generator);
		        			segmentIdx+=1;
		    			}
                    	coord_pair.x += x;
                    	coord_pair.y += y;
		    			offset += 1;
                	}
				}
	        	chnl += 1;
    	    }
        }
    }
};
