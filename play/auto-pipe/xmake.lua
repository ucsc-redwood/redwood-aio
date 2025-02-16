target("auto-pipe")
    set_kind("binary")
    add_files("main.cpp")

    add_packages("spdlog")
    -- add_packages("cli11")

    add_rules("run_on_android")