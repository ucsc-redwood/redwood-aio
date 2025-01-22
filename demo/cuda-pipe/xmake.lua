
add_requires("benchmark")
add_requires("cli11")
add_requires("concurrentqueue")

target("pipe")
    set_kind("binary")
    add_files("2-stage.cpp")
    add_packages("builtin-apps")
    add_packages("benchmark")

    add_packages("cli11")
    add_packages("concurrentqueue")
    add_packages("openmp")
target_end()



-- only available on non-android platforms
if not is_plat("android") then

target("pipe-cu")
    set_kind("binary")
    add_files("cu-stage.cpp")
    add_packages("builtin-apps")
    add_packages("benchmark")

    add_packages("cli11")
    add_packages("concurrentqueue")
    add_packages("openmp")

    add_files("cuda/*.cu")
    add_cugencodes("native")
target_end()


end

