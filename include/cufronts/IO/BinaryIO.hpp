#pragma once
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <cufronts/types/typedefs.hpp>
#include <netcdf>
#include <cufronts/IO/netCDFTypeConversion.hpp>

namespace fs = std::filesystem;


namespace frontIO{

    template<class T>
    using variableData = std::vector<T>;
    template<class container_t>
    using variableMap = std::map<std::string, container_t>;
    template<class T>
    using defaultVariableMap = variableMap<variableData<T>>;

    template<class T>
    using frontCoords = std::vector<val2<T>>;
    template<class T>
    using frontalTypeList = std::vector<frontCoords<T>>;
    template<class T>
    using frontMap = std::map<std::string, frontalTypeList<T>>;




    class BinaryReader
    {
    private:
        size_t getFileSize(fs::path filename){
            return fs::file_size(filename);
        }

    public:
        template<class T>
        std::vector<T> read(fs::path filename){
            auto sz = getFileSize(filename);    
            std::ifstream infile(filename, std::ios::binary);
            std::vector<T> data(sz/sizeof(T));
            infile.read((char*)data.data(), sz);
            infile.close();
            return data;
        }
        template<class T>
        void read(fs::path filename, std::vector<T>& data, bool resize = false){
            auto sz = getFileSize(filename);     
            std::ifstream infile(filename, std::ios::binary);
            if(resize) data.resize(sz/sizeof(T));
            infile.read((char*)data.data(), sz);
            infile.close();
        }
  
        template<class T>
        void write(fs::path filename, T data, size_t size = 0){
            std::ofstream outfile(filename, std::ios::binary);
            if(size == 0)
                outfile.write((char*)data.data(), data.size()*sizeof(typename T::value_type));
            else
                outfile.write((char*)data.data(), size*sizeof(typename T::value_type));
            outfile.close();
        }
        template<class T>
        void write(fs::path filename, T* data, size_t size){
            std::ofstream outfile(filename, std::ios::binary);
            outfile.write((char*)data, size*sizeof(T));
            outfile.close();
        }
    };




    class netCDFReader
    {
    private:
        /********
        // CREATE NEW BUFFER
        ***************/
        template<class T>
        variableData<T> readVariable(netCDF::NcFile& myFile, std::vector<size_t>& start, std::vector<size_t>& counts, const std::string& variable){
            auto var = myFile.getVar(variable);
            
            int total_size = 1;
            for(auto& count: counts){
                total_size*= count;
            }
            variableData<T> data(total_size);

            var.getVar(start, counts, data.data());
            
            T add_offset = 0;
            T scale_factor = 0;
            var.getAtt("add_offset").getValues(&add_offset);
            var.getAtt("scale_factor").getValues(&scale_factor);
            std::transform(data.begin(), data.end(), data.begin(), [&](T a){return a*scale_factor+add_offset;});
            return data;
        }

        /********
        // FILL EXISTING BUFFER
        ***************/
        template<class container_t>
        void readVariable(netCDF::NcFile& myFile, std::vector<size_t>& start, std::vector<size_t>& counts, container_t& data, const std::string& variable){
            auto var = myFile.getVar(variable);
            int total_size = 1;
            for(auto& count: counts){
                total_size*= count;
            }
            if(total_size != data.size())
                std::cerr << "WARNING: size of buffer ("<<data.size()<<") is smaller than requested size (" << total_size << ")" << std::endl;
            var.getVar(start, counts, thrust::raw_pointer_cast(data.data()));
            
            typename container_t::value_type add_offset = 0;
            typename container_t::value_type scale_factor = 0;
            var.getAtt("add_offset").getValues(&add_offset);
            var.getAtt("scale_factor").getValues(&scale_factor);
            std::transform(data.begin(), data.end(), data.begin(), [&](typename container_t::value_type a){return a*scale_factor+add_offset;});
        }


        template<class T>
        defaultVariableMap<T> read(netCDF::NcFile& myFile, const std::vector<std::string>& variables, std::vector<size_t>& start, std::vector<size_t>& counts){
            defaultVariableMap<T> results;
            for(const auto& variable: variables){
                results[variable] = readVariable<T>(myFile, start, counts, variable);
            }
            return results;
        }


        template<class container_t>
        void read(netCDF::NcFile& myFile, const std::vector<std::string>& variables, std::vector<size_t>& start, std::vector<size_t>& counts, variableMap<container_t>& results){
            for(const auto& variable: variables){
                readVariable(myFile, start, counts, results[variable], variable);
            }
        }




        std::vector<float> getAxis(netCDF::NcFile& myFile, std::string axisname){
            auto axVar = myFile.getVar(axisname);
            auto axDim = myFile.getDim(axisname);
            std::vector<float> axdat(axDim.getSize());
            axVar.getVar(axdat.data());
            return axdat;
        }

        // Converts Level, latitude and longitude into a 4d point corresponding to the indices 
        // within the data file
        template<class T>
        std::vector<size_t> convertCoordsToPix(netCDF::NcFile& myFile, T level, T latitude, T longitude){
            auto latdat = getAxis(myFile, "latitude");
            auto londat = getAxis(myFile, "longitude");
            auto levdat = getAxis(myFile, "level");
            
            // get lower Bound 
            auto latPos = std::lower_bound(latdat.rbegin(),latdat.rend(), latitude);
            size_t latIndex = std::distance(latdat.begin(), latPos.base())-1;
            auto lonPos = std::lower_bound(londat.begin(),londat.end(), longitude);
            size_t lonIndex = std::distance(londat.begin(), lonPos);
            auto levPos = std::lower_bound(levdat.begin(),levdat.end(), level);
            size_t levIndex = std::distance(levdat.begin(), levPos);
            std::vector<size_t> positions = {0,levIndex,latIndex, lonIndex};
            return positions; 
        }

        
    public:
        
        template<class T>
        defaultVariableMap<T> readCoordRange(fs::path filename, std::vector<std::string>& variables, std::vector<float>& startCoords, std::vector<float>& endCoords){
            netCDF::NcFile myFile(filename, netCDF::NcFile::read);
            auto startPix = convertCoordsToPix(myFile, startCoords[0], startCoords[1], startCoords[2]);
            // at this point we extract the position of the final pixel
            auto counts = convertCoordsToPix(myFile, endCoords[0], endCoords[1], endCoords[2]);
            // convert position to range
            for(int entry=0;entry<counts.size();++entry){
                counts[entry] = counts[entry] - startPix[entry] + 1;
            }
            auto result = read<T>(myFile, variables, startPix, counts);
            myFile.close();
            return result;
        }
        
        template<class container_t>
        void readCoordRange(fs::path filename, const std::vector<std::string>& variables, std::vector<float>& startCoords, std::vector<float>& endCoords, variableMap<container_t>& out_buffer){
            netCDF::NcFile myFile(filename, netCDF::NcFile::read, netCDF::NcFile::nc4classic);
            auto startPix = convertCoordsToPix(myFile, startCoords[0], startCoords[1], startCoords[2]);
            // at this point we extract the position of the final pixel
            auto counts = convertCoordsToPix(myFile, endCoords[0], endCoords[1], endCoords[2]);
            // convert position to range
            for(int entry=0;entry<counts.size();++entry){
                counts[entry] = counts[entry] - startPix[entry] + 1;
            }
            read(myFile, variables, startPix, counts, out_buffer);
            myFile.close();
        }
    };


    class netCDFWriter
    {   
    public:
        template<class container_t, class data_container_t>
        void write(fs::path filename, variableMap<data_container_t>& variables, variableMap<container_t>& axes, variableMap<std::vector<DTYPE_>>& packingParameter){
            netCDF::NcFile myFile(filename, netCDF::NcFile::replace);
            std::vector<netCDF::NcDim> dimArray;
            std::vector<netCDF::NcVar> varArray;
            dimArray.push_back(myFile.addDim("time",1));
            // set axes 
            dimArray.push_back(myFile.addDim("level", axes["level"].size()));
            dimArray.push_back(myFile.addDim("latitude", axes["latitude"].size()));
            dimArray.push_back(myFile.addDim("longitude", axes["longitude"].size()));
            // define axis variables
            for(auto& keyval: axes){
                typename netCDFTypeSelector<typename container_t::value_type>::nctype writeType;
                myFile.addVar(keyval.first, writeType, myFile.getDim(keyval.first));
            }

            // define 3d variables
            for(auto& keyval: variables){
                typename netCDFTypeSelector<typename data_container_t::value_type>::nctype writeType;
                auto var = myFile.addVar(keyval.first, writeType, dimArray);
                var.putAtt("add_offset", netCDF::ncFloat, packingParameter[keyval.first][0]);
                var.putAtt("scale_factor", netCDF::ncFloat, packingParameter[keyval.first][1]);
            }

            // write axis variables
            for(auto& keyval: axes){
                auto var = myFile.getVar(keyval.first);
                var.putVar(thrust::raw_pointer_cast(keyval.second.data()));
            }
            // write 3d variables
            for(auto& keyval: variables){
                auto var = myFile.getVar(keyval.first);
                var.putVar(thrust::raw_pointer_cast(keyval.second.data()));
            }
            myFile.close();
        }
    };

    


    class CSBReader
    {
    private:
        size_t getFileSize(fs::path filename){
            return fs::file_size(filename);
        }
        float lon_LeftEnd_;
        float lon_RightEnd_;
        float lat_TopEnd_;
        float lat_BottomEnd_;
        float lon_Step_;
        float lat_Step_;

    public:
        bool readToGrid_ = true;

        CSBReader(float lon_LeftEnd = -180, float lon_RightEnd = 180,
                    float lat_TopEnd = 90, float lat_BottomEnd = -90,
                    float lon_Step = 0.25,float lat_Step = -0.25 , 
                    bool readToGrid = true )
                    : lon_LeftEnd_(lon_LeftEnd), lon_RightEnd_(lon_RightEnd),
                    lat_TopEnd_(lat_TopEnd),lat_BottomEnd_(lat_BottomEnd),
                    lon_Step_(lon_Step),lat_Step_(lat_Step),
                    readToGrid_(readToGrid) {

                    }

        template<class T>
        frontMap<T> read(fs::path filename){
            auto sz = getFileSize(filename);     
            std::ifstream infile(filename, std::ios::binary);
            std::string line;
            // ignore first line
            std::getline(infile, line);
            frontMap<T> results;
            while(std::getline(infile, line)){
                parseLine(line, results);
            }
            infile.close();
            return results;
        }

        template<class T>
        void read(fs::path filename, frontMap<T>& results){
            auto sz = getFileSize(filename);     
            std::ifstream infile(filename, std::ios::binary);
            std::string line;
            // ignore first line
            std::getline(infile, line);
            while(std::getline(infile, line)){
                parseLine(line, results);
            }
            infile.close();
            return;
        }

        template<class T>
        void parseLine(std::string& line, frontMap<T>& results){
            // split at :
            std::string frontal_type = line.substr(0, line.find(":"));
            std::string coords = line.substr(line.find(":")+1);
            frontCoords<T> frontCoords; 
            // split at " "
            std::string delimiter = " ";
            std::string coordpair;
            T latcoord, loncoord, prevLoncoord, prevLatcoord;
            size_t posb = 0, pose = 0;
            auto previousCoordIsValid = false;
            auto previousCoordIsIncluded = true;
            while((pose = coords.find(delimiter,posb)) != std::string::npos){
                coordpair = coords.substr(posb,pose);
                auto splitPos = coordpair.find(",");
                latcoord = T(std::stof(coordpair.substr(0, splitPos)));
                loncoord = T(std::stof(coordpair.substr(splitPos+1)));
                if(readToGrid_){
                    latcoord = (latcoord - lat_TopEnd_)/lat_Step_;
                    loncoord = (loncoord - lon_LeftEnd_)/lon_Step_;
                    float latlim = (lat_BottomEnd_-lat_TopEnd_)/lat_Step_; 
                    float lonlim = (lon_RightEnd_-lon_LeftEnd_)/lon_Step_;
                    // with restricted grid read => ignore all values outside of grid
                    bool hasValidPosition = latcoord >= 0 && loncoord >= 0 && latcoord < latlim && loncoord < lonlim;
                    if(hasValidPosition){
                        if(!previousCoordIsIncluded)
                            frontCoords.push_back(val2<T>(prevLoncoord, prevLatcoord));    
                        // skip the case where two consecutive points fall onto the same position
                        // in this case the interpolation would be useless anyway
                        
                        if(prevLoncoord != loncoord || prevLatcoord != latcoord){
                            frontCoords.push_back(val2<T>(loncoord, latcoord));
                        }
                        
                        previousCoordIsValid = true;
                        previousCoordIsIncluded = true;
                    }
                    else if(!hasValidPosition && previousCoordIsValid){
                        frontCoords.push_back(val2<T>(loncoord, latcoord));
                        previousCoordIsValid = false;
                        previousCoordIsIncluded = true;
                    }
                    else{
                        previousCoordIsValid = false;
                        previousCoordIsIncluded = false;
                    }
                    prevLoncoord = loncoord;
                    prevLatcoord = latcoord;
                }
                else{
                    frontCoords.push_back(val2<T>(loncoord, latcoord));
                }
                posb = pose+1;
            }
            if(frontCoords.size()>1){
                results[frontal_type].push_back(frontCoords);
            }
        }

        template<class T>
        void write(fs::path filename, T data){
            std::ofstream outfile(filename, std::ios::binary);
            outfile.write((char*)data.data(), data.size()*sizeof(typename T::value_type));
            outfile.close();
        }
    };
};
