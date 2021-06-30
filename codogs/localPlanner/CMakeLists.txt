project(cppGen)
cmake_minimum_required(VERSION 3.7)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

link_directories(/usr/local/lib)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})
add_subdirectory(generated)    #localPlan

add_library(localPlanner SHARED localPlanner.cpp tonlp.cpp )
target_link_libraries(localPlanner ipopt)
target_link_libraries (localPlanner Eigen3::Eigen)
target_link_libraries(localPlanner localPlan)
target_include_directories(localPlanner PRIVATE /usr/local/include/coin-or)


add_executable(testLocalPlanner testLocalPlanner.cpp)
target_link_libraries(testLocalPlanner localPlanner)
target_link_libraries(testLocalPlanner Eigen3::Eigen)
