target("try-sort")
    add_rules("vulkan_config", "run_on_android")
    set_kind("binary")
    add_files("main.cpp")
    
    add_includedirs("$(projectdir)")
    add_deps("builtin-apps-vulkan")
    add_packages("spdlog")
target_end()