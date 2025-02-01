add_requires("volk")

target("query-warpsize")
    set_kind("binary")
    set_group("utility")
    add_files("query_warpsize.cpp")
    
    add_packages("vulkan-hpp")
    add_packages("volk")

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()

