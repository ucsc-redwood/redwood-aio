# #!/usr/bin/env fish
# # Define the header (no newline needed; added in sed command)
# set header "-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License."

# # Find all .lua files and update each one
# for file in (find . -type f -name '*.lua')
#     sed -i "1s|^|$header\n|" $file
#     echo "Updated: $file"
# end

#!/usr/bin/env fish
# For each .lua file found under the current directory:
for file in (find . -type f -name '*.lua')
    # Using GNU sed's insert command to add two lines at the top:
    sed -i '1i\
-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.\
' $file
    echo "Updated: $file"
end
