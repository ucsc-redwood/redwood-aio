if not is_plat("android") then
    -- set_toolchains("clang")
    add_requires("openmp")
end

add_requires("spdlog")

-- CPU
target("builtin-apps")
    set_kind("static")
    add_headerfiles("**/*.hpp")
    add_files("**/*.cpp")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("spdlog")
target_end()


for _, file in ipairs(os.files("**/test_*.cpp")) do
     local name = path.basename(file)
     target(name)
         set_kind("binary")
         set_default(false)
         add_files(file)
         
         add_headerfiles("**/*.hpp")
         add_files("**/*.cpp")

         add_tests("default")
         add_tests("default_output", {trim_output = true, pass_outputs = "good"})
end

