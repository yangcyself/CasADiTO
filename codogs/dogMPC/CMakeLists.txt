project(cppGen)
cmake_minimum_required(VERSION 3.7)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

link_directories(/usr/local/lib)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})
add_subdirectory(generated)    #localPlan

add_library(dogController SHARED dogController.cpp tonlp.cpp )
target_link_libraries(dogController ipopt)
target_link_libraries (dogController Eigen3::Eigen)
target_link_libraries(dogController localPlan)
target_include_directories(dogController PRIVATE /usr/local/include/coin-or)


add_executable(testdogController testdogController.cpp)
target_link_libraries(testdogController dogController)
target_link_libraries(testdogController Eigen3::Eigen)
