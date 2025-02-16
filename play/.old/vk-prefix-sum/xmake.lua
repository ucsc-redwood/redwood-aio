-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("vk-prefix-sum")
    set_kind("binary")
    set_group("play")
    add_files("single.cpp")

    add_deps("builtin-apps")

    add_includedirs("$(projectdir)/builtin-apps/")

    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("spdlog")
    -- add_packages("cli11")

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()

