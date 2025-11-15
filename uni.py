#!/usr/bin/env python3

import sys
import os

# Define the characters to replace and what to replace them with.
# \xa0 is the common Non-Breaking Space (often seen as a syntax error in PHP/Python).
# \u200b is the Zero-Width Space.
REPLACEMENTS = {
    '\xa0': ' ',  # Replace Non-Breaking Space with a standard space
    '\u200b': '', # Remove Zero-Width Space entirely
    '\ufeff': '', # Remove Byte Order Mark (BOM), common in Windows-saved UTF-8 files
}

def clean_file_characters(file_path):
    """
    Reads a file, replaces specific problematic characters, and overwrites the file.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at path: {file_path}")
        return

    print(f"INFO: Attempting to clean file: {file_path}")

    # Step 1: Read the file content
    try:
        # Read the file using standard UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError as e:
        print(f"ERROR: Failed to read file due to Unicode error. Check file encoding. Error: {e}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during reading: {e}")
        return

    # Step 2: Perform replacements
    new_content = content
    for old_char, new_char in REPLACEMENTS.items():
        count = new_content.count(old_char)
        if count > 0:
            new_content = new_content.replace(old_char, new_char)
            print(f"INFO: Replaced {count} instances of U+{ord(old_char):04X} ('{old_char.strip()}') with '{new_char.strip() if new_char else '[deleted]'}'.")

    # If no changes were made, skip writing
    if new_content == content:
        print("INFO: No problematic characters found. File remains unchanged.")
        return

    # Step 3: Overwrite the original file with the cleaned content
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("SUCCESS: File cleaned and overwritten successfully.")
    except Exception as e:
        print(f"ERROR: Failed to write to file: {e}")

if __name__ == "__main__":
    # Ensure the script receives exactly one argument (the file path)
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_php_file>")
        sys.exit(1)

    # The file path is the second item in the argument list
    target_file = sys.argv[1]
    clean_file_characters(target_file)
