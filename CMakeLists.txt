cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11) 
project( FTT )

if(${CMAKE_CURRENT_SOURCE_DIR} MATCHES "/home/jamesa/+")
    SET(OpenCV_DIR "/home/miko/opencv3/lib/share/OpenCV/")
endif()

find_package( OpenCV REQUIRED )

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -march=native") 
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -march=native")
endif()

include_directories("${PROJECT_BINARY_DIR}")
add_executable( FTT ForestTrainingTesting.cpp
					Classification.cpp
					DataPointCollection.cpp
					FeatureResponseFunctions.cpp
					IPUtils.cpp
					Regression.cpp
					StatisticsAggregators.cpp )
target_link_libraries( FTT ${OpenCV_LIBS} )
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

