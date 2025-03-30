for i in range(1, 10):
    for j in range(i, 10):
        print(f"void run_stage_{i}_{j}(cifar_dense::AppData &data) {{")
        for k in range(i, j + 1):
            print(f"    cifar_dense::omp::process_stage_{k}(data);")
        print("}\n")

