# states the cmake version requirement
cmake_minimum_required(VERSION 3.10)

# set the project name and MAJOR.MINOR.PATCH version number
project(simple_test VERSION 1.1.0)

# add the executable
add_executable(simple_test_exe test.cpp)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)