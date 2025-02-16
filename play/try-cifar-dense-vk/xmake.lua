-- ----------------------------------------------------------------
-- Playground: Testing CIFAR Dense Network with Vulkan
-- ----------------------------------------------------------------

target("try-cifar-dense-vk") do
    add_rules("vulkan_config", "common_flags", "run_on_android")
    set_kind("binary")
    add_files({
        "main.cpp",
    })
    
    add_includedirs("$(projectdir)")
    add_deps("builtin-apps", "builtin-apps-vulkan")
    add_packages("spdlog")
end
