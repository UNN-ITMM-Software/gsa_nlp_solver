cmake_minimum_required(VERSION 3.2)

set (CMAKE_CXX_STANDARD 11)

option(BUILD_BINDINGS "Build python bindings" OFF)

macro(subdirlist result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child} AND NOT (${child} MATCHES "[.]"))
        list(APPEND dirlist ${curdir}/${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

set(PROJECT_NAME_STR go_problems)

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/go-problems)
set(PROBLEMS_LIBS "")
foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(GO_INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/go-problems)
else()
  set(GO_INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/go-problems PARENT_SCOPE)
endif()

if (${BUILD_BINDINGS})
  if (NOT TARGET pybind11)
    add_subdirectory(3rd-party/pybind11)
  endif()
  pybind11_add_module(go_problems ${CMAKE_CURRENT_SOURCE_DIR}/go-problems/bindings.cpp)
  target_include_directories(go_problems PUBLIC ${GO_INTERFACE_INCLUDE_DIR}/)
  target_link_libraries(go_problems PUBLIC ${PROBLEMS_LIBS})
endif()
