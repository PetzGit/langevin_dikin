add_executable(cldiff_tests
    test_black_scholes.cpp
    test_cos_pricer.cpp
    test_heston_objective.cpp
)
target_link_libraries(cldiff_tests PRIVATE cldiff Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(cldiff_tests)
