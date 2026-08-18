add_executable(vectorforge_tests
    test_distance.cpp
    test_flat.cpp
    test_serialization.cpp
    test_hnsw.cpp
    test_hnsw_serialization.cpp
)

target_link_libraries(vectorforge_tests
    PRIVATE
        vectorforge::vectorforge
        Catch2::Catch2WithMain
)

if(NOT MSVC)
    target_compile_options(vectorforge_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(Catch)
catch_discover_tests(vectorforge_tests PROPERTIES TIMEOUT 300)
