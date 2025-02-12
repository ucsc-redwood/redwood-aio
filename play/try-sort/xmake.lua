target("try-sort")
    set_kind("binary")
    add_files("main.cpp")
    
    add_includedirs("$(projectdir)")

    add_deps("builtin-apps-vulkan")
    
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("spdlog")

    if is_plat("android") then
      on_run(run_on_android)
    end
