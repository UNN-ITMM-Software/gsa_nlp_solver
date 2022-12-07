cmake_minimum_required(VERSION 3.5)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_BINDINGS "Build python bindings" OFF)
if (${BUILD_BINDINGS})
  add_subdirectory(3rd-party/pybind11)
endif()

option(GSA_USE_OPEN_CV "Set ON to use OpenCV" OFF)
set(GSA_OPEN_CV_PATH "" CACHE STRING "Path to OpenCV build")

if (${GSA_USE_OPEN_CV})
  find_package( OpenCV REQUIRED )
  include_directories(  )
  set(GSA_OPEN_CV_LIB ${OpenCV_LIBS})
  set(GSA_OPEN_CV_INCLUDES ${OpenCV_INCLUDE_DIRS} )
#  set(GSA_OPEN_CV_LIB_DIR "")
  
  add_definitions(-DUSE_OpenCV)
else()
  set(GSA_OPEN_CV_INCLUDES "")
  set(GSA_OPEN_CV_LIB_DIR "")
  set(GSA_OPEN_CV_LIB "")
endif()

include_directories(
  ${GSA_OPEN_CV_INCLUDES})

link_directories(
  ${GSA_OPEN_CV_LIB_DIR})

add_subdirectory(3rd-party/global-optimization-test-problems)
set(CMDLINE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/3rd-party/cmdline)

add_subdirectory(solver)
add_subdirectory(samples)
