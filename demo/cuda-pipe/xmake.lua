
add_requires("benchmark")
add_requires("cli11")
add_requires("concurrentqueue")

-- only available on non-android platforms
if not is_plat("android") then
target("cuda-pipe")
    set_kind("binary")
    add_files("*.cpp")
    add_packages("builtin-apps")
    add_packages("benchmark")

    add_packages("cli11")
    add_packages("concurrentqueue")
    add_packages("openmp")
target_end()
end
