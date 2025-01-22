add_requires("cli11")

add_requires("concurrentqueue")

add_requires("spdlog")


target("vk")
    set_kind("binary")
    add_files("main.cpp")
    add_files("../../builtin-apps/cifar-dense/vulkan/vk_dispatcher.cpp")


    add_deps("builtin-apps")
    add_deps("vk-backend")

    add_packages("concurrentqueue")
    add_packages("cli11")
    add_packages("spdlog")

    add_packages("vulkan-hpp", "vulkan-memory-allocator")

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()

