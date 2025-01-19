add_rules("mode.debug", "mode.release")

if not is_plat("android") then
    set_toolchains("clang")
    add_requires("openmp")
end

set_languages("c++20")
set_warnings("allextra")

add_requires("spdlog")

target("builtin-apps")
    set_kind("static")
    add_includedirs("includes")
    add_headerfiles("includes/**/*.hpp")
    add_headerfiles("includes/*.hpp")
    add_files("src/**/*.cpp")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("spdlog")

