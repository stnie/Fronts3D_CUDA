
set(HEADER_LIST "${CuFront_SOURCE_DIR}/include/cufronts/IO/*.cuh")

###########################
# - A Library of all the headers
add_library(IO_library_interface INTERFACE)


###################################################################################################
# - define target   -------------------------------------------------------------------------------


target_include_directories(IO_library_interface INTERFACE 
                            "{CuFront_SOURCE_DIR}/include" ,
                            "${netcdf_cxx4_SOURCE_DIR}/cxx4" ,
                            "${argparse_SOURCE_DIR}/include/" )

#link_directories()
target_link_libraries(IO_library_interface INTERFACE netcdf netcdf-cxx4)


source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST}
)

