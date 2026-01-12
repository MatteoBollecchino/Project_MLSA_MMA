import os
import shutil
import datetime
import re

# --- Configuration ---
PROJECT_DIR_NAME = "Project"
EXPORT_BASE_DIR_NAME = "Pro_Exp"
OLD_VERSIONS_DIR_NAME = "old"
ZIP_FILENAME_BASE = "Project_MLSA_Pietri_Bollecchino_Nesti"
# --- End Configuration ---

def main():
    """
    Archives the project folder into a versioned zip file, managing old versions.
    """
    try:
        # Get the absolute path of the script's directory (TOOLS)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the absolute path of the parent directory of TOOLS (the project root)
        project_root_dir = os.path.dirname(script_dir)

        # --- Setup Paths ---
        project_dir_to_zip = os.path.join(project_root_dir, PROJECT_DIR_NAME)
        export_dir = os.path.join(project_root_dir, EXPORT_BASE_DIR_NAME)
        old_versions_dir = os.path.join(export_dir, OLD_VERSIONS_DIR_NAME)

        # --- Create Directories ---
        # The script creates these folders if they don't exist, as requested.
        print(f"Ensuring export directory exists: {export_dir}")
        os.makedirs(export_dir, exist_ok=True)
        print(f"Ensuring old versions directory exists: {old_versions_dir}")
        os.makedirs(old_versions_dir, exist_ok=True)
        
        if not os.path.isdir(project_dir_to_zip):
            print(f"Error: The directory to zip '{project_dir_to_zip}' does not exist.")
            return

        # --- Determine Next Version ---
        highest_version = 0
        all_zips = []
        for dirpath, _, filenames in os.walk(export_dir):
            for filename in filenames:
                if filename.endswith('.zip') and ZIP_FILENAME_BASE in filename:
                    all_zips.append(filename)

        for filename in all_zips:
            match = re.search(r'_V(\d+)\.zip$', filename)
            if match:
                version = int(match.group(1))
                if version > highest_version:
                    highest_version = version
        
        next_version = highest_version + 1
        print(f"Determined next version: V{next_version}")

        # --- Move Existing Latest Zip to 'old' Folder ---
        # We only look in the top-level of the export directory.
        for item in os.listdir(export_dir):
            if item.endswith('.zip') and os.path.isfile(os.path.join(export_dir, item)):
                src_path = os.path.join(export_dir, item)
                dest_path = os.path.join(old_versions_dir, item)
                print(f"Moving existing archive '{item}' to '{old_versions_dir}'")
                shutil.move(src_path, dest_path)

        # --- Create New Zip File ---
        current_date = datetime.datetime.now().strftime("%d%m%Y")
        zip_filename_stem = f"{ZIP_FILENAME_BASE}_{current_date}_V{next_version}"
        zip_filepath_stem = os.path.join(export_dir, zip_filename_stem)

        print(f"Creating new archive for '{PROJECT_DIR_NAME}'...")
        
        # shutil.make_archive creates an archive of files in 'base_dir'.
        # 'root_dir' is the directory that will be the root of the archive.
        # All paths in the archive will be relative to 'root_dir'.
        archive_path = shutil.make_archive(
            base_name=zip_filepath_stem,
            format='zip',
            root_dir=project_root_dir,
            base_dir=PROJECT_DIR_NAME
        )

        print(f"Successfully created '{os.path.basename(archive_path)}' in '{export_dir}'.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
