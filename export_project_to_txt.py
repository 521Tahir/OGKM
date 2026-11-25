import os

PROJECT_DIR = "/home/tahirahmad/Ontology_Project"
OUTPUT_FILE = "/home/tahirahmad/Ontology_Project/project_full_code.txt"

# File extensions commonly containing source code/configuration
INCLUDE_ALL_FILES = True

# Skip large/generated directories
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache"
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    outfile.write(f"PROJECT EXPORT: {PROJECT_DIR}\n")
    outfile.write("=" * 120 + "\n\n")

    for root, dirs, files in os.walk(PROJECT_DIR):
        # Remove excluded directories from traversal
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)

            # Skip the output file itself
            if os.path.abspath(file_path) == os.path.abspath(OUTPUT_FILE):
                continue

            rel_path = os.path.relpath(file_path, PROJECT_DIR)

            outfile.write("\n")
            outfile.write("=" * 120 + "\n")
            outfile.write(f"FILE: {rel_path}\n")
            outfile.write("=" * 120 + "\n\n")

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"\n[ERROR READING FILE: {e}]\n")

            outfile.write("\n\n")

print(f"Project exported successfully to: {OUTPUT_FILE}")