#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 source_file_list.txt source_directory target_directory"
    exit 1
fi

source_file_list="$1"
source_directory="$2"
target_directory="$3"

# Check if the source file list exists
if [ ! -f "$source_file_list" ]; then
    echo "Source file list does not exist: $source_file_list"
    exit 1
fi

# Check if the source directory exists
if [ ! -d "$source_directory" ]; then
    echo "Source directory does not exist: $source_directory"
    exit 1
fi

# Check if the target directory exists
if [ ! -d "$target_directory" ]; then
    echo "Target directory does not exist: $target_directory"
    exit 1
fi

# Loop through each line in the source file list and copy the files
while IFS= read -r file; do
    file="${file%"${file##*[![:space:]]}"}"  # Remove trailing whitespace

    source_file="$source_directory/$file"

    if [ -f "$source_file" ]; then
        cp "$source_file" "$target_directory"
        echo "Copied $source_file to $target_directory"
    else
        echo "File not found: $source_file"
    fi
done < "$source_file_list"

echo "Copy process completed."
