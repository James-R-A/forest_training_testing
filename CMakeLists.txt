cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11) 
project( FTT )

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

