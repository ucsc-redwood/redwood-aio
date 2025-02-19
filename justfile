
set-android:
    xmake f -p android -a arm64-v8a --ndk=~/Android/Sdk/ndk/28.0.13004108/ --android_sdk=~/Android/Sdk/ --ndk_sdkver=28 -c -v

set-jetson:
    xmake f -p linux -a arm64 -c -v --cuda=yes

set-default:
    xmake f -p linux -a x86_64 -c

push-all-resources:
    xmake push-all-resources

# bm-to-db:
#     rm -f data/benchmark_results.db
#     python3 scripts/database/run_splite_raw.py data/raw_logs/02_16_2025.txt
#     python3 scripts/database/run_insert_db.py data/raw_logs/3A021JEHN02756.txt
#     python3 scripts/database/run_insert_db.py data/raw_logs/ce0717178d7758b00b7e.txt
#     python3 scripts/database/run_insert_db.py data/raw_logs/9b034f1b.txt

db-to-code:
    python3 scripts/analysis/gen_schdules.py --device 3A021JEHN02756 --app CifarDense --output_dir ./data/generated-schedules/
    python3 scripts/analysis/gen_schdules.py --device 3A021JEHN02756 --app CifarSparse --output_dir ./data/generated-schedules/
    python3 scripts/analysis/gen_schdules.py --device 3A021JEHN02756 --app Tree --output_dir ./data/generated-schedules/
    
    python3 scripts/analysis/gen_schdules.py --device 9b034f1b --app CifarDense --output_dir ./data/generated-schedules/
    python3 scripts/analysis/gen_schdules.py --device 9b034f1b --app CifarSparse --output_dir ./data/generated-schedules/
    python3 scripts/analysis/gen_schdules.py --device 9b034f1b --app Tree --output_dir ./data/generated-schedules/
    
    python3 scripts/analysis/gen_schdules.py --device ce0717178d7758b00b7e --app CifarDense --output_dir ./data/generated-schedules/
    python3 scripts/analysis/gen_schdules.py --device ce0717178d7758b00b7e --app CifarSparse --output_dir ./data/generated-schedules/
    python3 scripts/analysis/gen_schdules.py --device ce0717178d7758b00b7e --app Tree --output_dir ./data/generated-schedules/

    python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/cifar-dense-vk/generated-code/ --application CifarDense
    python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/cifar-sparse-vk/generated-code/ --application CifarSparse
    python3 scripts/analysis/gen_pipes.py --in_dir ./data/generated-schedules/ --out_dir pipe/tree-vk/generated-code/ --application Tree
    
    xmake format