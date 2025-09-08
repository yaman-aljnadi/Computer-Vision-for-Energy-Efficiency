import os

def rename_images_in_folder(folder_path):
    # Get list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort files (optional: ensures consistent ordering)
    files.sort()

    for idx, filename in enumerate(files, start=1):
        # Extract the file extension
        ext = os.path.splitext(filename)[1]

        # Construct the new file name
        new_name = f"{idx}{ext}"

        # Construct full file paths
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(src, dst)
        print(f"Renamed '{filename}' to '{new_name}'")

# Example usage:
folder_path = "H:/Shared drives/Computer Vision for Energy Efficiency/Yaman/Custom_Created_Images/Adnan_Custome_Code_For_Images/Images"
rename_images_in_folder(folder_path)
