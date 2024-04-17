import os
import shutil

def copy_files_from_list(file_list_path, source_dir, output_dir):
    # Check if the source and output directories exist
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist.")
        return

    # Read the list of filenames from the input file
    with open(file_list_path, 'r') as file:
        file_names = [line.strip().strip(',') for line in file.readlines()]

    # Iterate over the file names and copy each file
    for file_name in file_names:
        source_path = os.path.join(source_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        if os.path.exists(source_path):
            shutil.copy2(source_path, output_path)
            print(f"Copied '{file_name}' to '{output_dir}'.")
        else:
            print(f"File '{file_name}' not found in source directory.")

if __name__ == "__main__":
    file_list_path = input("Enter the path to the text file containing file names: ")
    source_dir = input("Enter the path to the source directory: ")
    output_dir = input("Enter the path to the output directory: ")

    copy_files_from_list(file_list_path, source_dir, output_dir)
