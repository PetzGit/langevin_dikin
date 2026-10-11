cmake_minimum_required(VERSION 3.14)
project(cldiff LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies, fetched at configure time ---
include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
# Eigen ships its own CMakeLists.txt that builds tests/docs we don't want;
# these two lines skip that and just make the headers available.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(catch2)

# --- Your library ---
add_library(cldiff
    src/optim/polyhedron.cpp
    src/core/rng.cpp
    src/optim/samplers/dikin_langevin.cpp
    src/optim/objective.cpp
    src/optim/optimisers/annealed_langevin.cpp
    src/optim/schedules/three_phase_log.cpp
    src/optim/schedules/parallel_floor.cpp
    #src/optim/geometry/kinetic_dikin.cpp
    #src/optim/tempered_objective.cpp
    #src/optim/samplers/akdl.cpp
    #src/optim/optimisers/annealed_akdl.cpp
    src/models/heston/characteristic_function.cpp
    src/models/heston/cumulants.cpp
    src/pricing/fourier/cos_pricer.cpp
    src/models/black_scholes/black_scholes.cpp
    src/market/market_data.cpp
    src/models/heston/heston_cos_pricer.cpp
    src/calibration/calibration_objective.cpp
    src/calibration/implied_vol_squared_loss.cpp
    src/calibration/feller_penalty.cpp
    src/calibration/heston_objective.cpp
    src/models/heston/feller.cpp
    src/calibration/corrected_log_space_heston_objective.cpp
)
target_include_directories(cldiff PUBLIC include)
target_link_libraries(cldiff PUBLIC Eigen3::Eigen)

# Warnings + sanitizers, as discussed early on
target_compile_options(cldiff PRIVATE -Wall -Wextra -Werror)

# --- Tests ---
#enable_testing()
#add_subdirectory(tests)

add_subdirectory(experiments)
