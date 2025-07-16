import os
import shutil

# Extensions or filenames to delete
FILES_TO_DELETE = [".DS_Store"]
EXTENSIONS_TO_DELETE = [".pyc", ".pyo"]

# Folder names to delete
FOLDERS_TO_DELETE = ["__pycache__", ".ipynb_checkpoints"]  # remove ".vscode" if you want to keep settings

def should_delete_file(file):
    return file in FILES_TO_DELETE or any(file.endswith(ext) for ext in EXTENSIONS_TO_DELETE)

def clean_directory(root_path="."):
    deleted_files = 0
    deleted_dirs = 0

    for root, dirs, files in os.walk(root_path):
        # Delete matching files
        for file in files:
            if should_delete_file(file):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"🗑️  Deleted file: {file_path}")
                deleted_files += 1

        # Delete matching folders
        for dir in dirs:
            if dir in FOLDERS_TO_DELETE:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path, ignore_errors=True)
                print(f"🧹 Deleted folder: {dir_path}")
                deleted_dirs += 1

    print(f"\n✅ Cleanup complete: {deleted_files} files, {deleted_dirs} folders deleted.")

if __name__ == "__main__":
    clean_directory()
