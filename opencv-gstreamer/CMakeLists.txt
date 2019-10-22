cmake_minimum_required(VERSION 3.5)

project(inference_samples)

find_package(OpenCV 3.3.0 REQUIRED)
find_package(Boost 1.6.5 REQUIRED COMPONENTS program_options filesystem)

# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
# find_package(GStreamer REQUIRED)

# find_package(PkgConfig REQUIRED)
# pkg_search_module(GLIB REQUIRED glib-2.0)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library

set(GST_STREAM gst_stream)
add_library(${GST_STREAM} SHARED
    src/gst_stream.cpp
)
target_include_directories(${GST_STREAM} 
PUBLIC 
    include
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    # ${GSTREAMER_INCLUDE_DIRS}
    # ${GLIB_INCLUDE_DIRS}
)
target_link_libraries(${GST_STREAM} 
PUBLIC     
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES}
)


set(GST_STREAM_TEST gst-test)
add_executable(${GST_STREAM_TEST} 
    src/main.cpp
)
target_link_libraries(${GST_STREAM_TEST} 
    ${GST_STREAM}
)
target_include_directories(${GST_STREAM_TEST} 
PRIVATE
    include
)


