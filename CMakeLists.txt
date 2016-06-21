cmake_minimum_required(VERSION 2.8)
project( FTT )

# Macro to set c++ version to C++ 11. 
# Supports older versions of CMAKE which don't
# have set(CMAKE_CXX_STANDARD 11) functionality
# only applicable for GNU
macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_COMPILER_IS_GNUCC)
      set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

use_cxx11()

# if on the noisycat machine, this is where OPENCV is installed
# super hacky way of detecting what machine we're on
if(${CMAKE_CURRENT_SOURCE_DIR} MATCHES "/home/jamesa/+")
    SET(OpenCV_DIR "/home/miko/opencv3/lib/share/OpenCV/")
endif()

find_package( OpenCV REQUIRED )

find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -march=native") 
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -march=native")
endif()

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3") 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") 
endif()

include_directories("${PROJECT_BINARY_DIR}")
add_executable( FTT ForestTrainingTesting.cpp
			DataPointCollection.cpp
			FeatureResponseFunctions.cpp
			IPUtils.cpp
			StatisticsAggregators.cpp )

target_link_libraries( FTT ${OpenCV_LIBS} )

# Displays all available variables
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

