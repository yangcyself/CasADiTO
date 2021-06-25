project(cppGen)
cmake_minimum_required(VERSION 3.7)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC") # must set the release flag, or it will add -O3 by default

link_directories(/usr/local/lib)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})

add_subdirectory(generated)    # hrl

add_library(hrlDyn SHARED hrlDyn.cpp tonlp.cpp )
target_link_libraries(hrlDyn ipopt)
target_link_libraries (hrlDyn Eigen3::Eigen)
target_link_libraries(hrlDyn hrl)
target_include_directories(hrlDyn PRIVATE /usr/local/include/coin-or)


add_executable(testhrl testHrl.cpp)
target_link_libraries(testhrl hrlDyn)


# target_link_libraries (testhrl Eigen3::Eigen)
# target_link_libraries(testhrl ipopt)
# target_include_directories(testhrl PRIVATE /usr/local/include/coin-or /usr/local/include/coin-or/mumps)
