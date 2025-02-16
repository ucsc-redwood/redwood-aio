add_requires("volk")

target("query-warpsize")
    set_kind("binary")
    set_group("utility")
    add_files("query_warpsize.cpp")
    
    add_packages("vulkan-hpp")
    add_packages("volk")
    add_rules("run_on_android")

target_end()

if is_plat("linux") or is_plat("android") then
target("query-cpuinfo")
    set_kind("binary")
    set_group("utility")
    add_files("query_cpuinfo.cpp")


    add_rules("run_on_android")
target_end()
end

target("test-affinity")
    set_kind("binary")
    set_group("utility")
    add_files("test_affinity.cpp")

    add_rules("run_on_android")
target_end()

target("query-cacheline") 
    set_kind("binary")
    set_group("utility")
    add_files("query_cacheline.cpp")

    add_rules("run_on_android")
target_end()
