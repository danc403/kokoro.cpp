#!/usr/bin/env python3
import numpy as np
import json
import argparse
import sys
import os
from typing import Dict, Any, Optional
import re 
import struct # Added for binary packing

# Define the default voice file path
DEFAULT_VOICE_FILE = "voices-v1.0.bin"

def get_gender_from_prefix(voice_name: str) -> str:
    # Function body is unchanged
    try:
        prefix = voice_name.split('_')[0].lower()
        if not prefix:
            return 'Unknown'
        
        last_char = prefix[-1]
        
        if last_char == 'f':
            return 'Female'
        elif last_char == 'm':
            return 'Male'
        else:
            return 'Unknown'
            
    except Exception:
        return 'Unknown'

def clean_voice_array(voice_array: np.ndarray) -> np.ndarray:
    # Function body is unchanged
    return np.squeeze(voice_array)

def get_clean_2d_array(data: np.ndarray) -> np.ndarray:
    # Function body is unchanged
    cleaned_data = np.squeeze(data)
    
    if cleaned_data.ndim == 1:
        if cleaned_data.size == 0:
             return cleaned_data.reshape(0, 0)
        return cleaned_data.reshape(-1, 1)
    
    elif cleaned_data.ndim > 2:
        if cleaned_data.size == 0:
             return cleaned_data.reshape(0, 0)

        new_rows = np.prod(cleaned_data.shape[:-1])
        last_dim = cleaned_data.shape[-1]
        return cleaned_data.reshape(new_rows, last_dim)
        
    return cleaned_data


def load_voice_data(filepath: str, suppress_output: bool = False) -> Optional[Dict[str, np.ndarray]]:
    # Function body is largely unchanged, prints redirected to stderr if not suppressed
    if not os.path.exists(filepath):
        print(f"Error: Input file not found at '{filepath}'.", file=sys.stderr)
        return None

    if filepath.lower().endswith(('.json')):
        if not suppress_output:
            print(f"Loading data from JSON file: '{filepath}'...", file=sys.stderr)
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            voice_map = {}
            if isinstance(json_data, dict):
                
                source_dict = json_data
                
                for k, v in source_dict.items():
                    if k == 'voices_array' or k == 'voice_name' or k == 'data':
                        continue
                        
                    try:
                        data_array = np.array(v, dtype=np.float32)
                        voice_map[k] = clean_voice_array(data_array) 
                    except Exception as e:
                        if not suppress_output:
                            print(f"Error converting data for key '{k}'. Skipped. Details: {e}", file=sys.stderr)

            elif isinstance(json_data, list):
                if not suppress_output:
                    print("Warning: Loaded old single-voice JSON format. Relying on filename.", file=sys.stderr)
                filename = os.path.basename(filepath)
                voice_name = os.path.splitext(filename)[0]
                data_array = np.array(json_data, dtype=np.float32)
                voice_map[voice_name] = clean_voice_array(data_array)
                
            return voice_map
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{filepath}'. Details: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error processing JSON data from '{filepath}'. Details: {e}", file=sys.stderr)
            return None
            
    elif filepath.lower().endswith(('.bin', '.npy', '.npz')):
        if not suppress_output:
            print(f"Loading data from NumPy binary file: '{filepath}'...", file=sys.stderr)
        try:
            data = np.load(filepath, allow_pickle=True)
            return {k: clean_voice_array(data[k]) for k in data.files}
        except Exception as e:
            print(f"Error loading NumPy data from '{filepath}'. Details: {e}", file=sys.stderr)
            return None
    else:
        print(f"Error: Unsupported file type for '{filepath}'. Must be .bin, .npy, .npz, or .json.", file=sys.stderr)
        return None

# --- New Function for Raw Binary Export ---

def save_voice_to_raw_binary(voice_array: np.ndarray, voice_name: str):
    """
    Exports a single voice's vectors to a raw binary file (.dat) 
    with a simple [int N, int D] header.
    """
    voice_array_2d = get_clean_2d_array(voice_array)
    
    if voice_array_2d.ndim != 2:
        print(f"Error: Cannot save '{voice_name}' to binary; data is not 2D ({voice_array_2d.shape}).", file=sys.stderr)
        return False
        
    rows, cols = voice_array_2d.shape
    output_filename = f"{voice_name}.dat"
    
    try:
        with open(output_filename, 'wb') as f:
            # 1. Write Header: [int rows (N), int cols (D)]
            # Use 'i' for standard 4-byte integers
            f.write(struct.pack('<2i', rows, cols)) 
            
            # 2. Write Data: Raw float32 data
            # Flatten the 2D array and convert to raw bytes
            f.write(voice_array_2d.astype(np.float32).tobytes())
            
        print(f"✅ Successfully exported '{voice_name}' to '{output_filename}' (Raw Binary).")
        return True
        
    except Exception as e:
        print(f"Error exporting raw binary file for '{voice_name}': {e}", file=sys.stderr)
        return False

# --- Modified extract_voices function ---

def extract_voices(voice_map: Dict[str, np.ndarray], voice_name: str, input_is_json: bool):
    """
    Extracts voices to JSON, and if it's a single voice, also extracts to Raw Binary (.dat).
    """
    if input_is_json and not voice_name:
        print("✅ Skipping extraction of 'all' voices as the input was a JSON file.")
        return
        
    if not voice_name:
        # Extract ALL voices to a single JSON file (No binary output here)
        print(f"Extracting ALL {len(voice_map)} voices to 'voices.json'...")
        
        output_data = {k: get_clean_2d_array(v).tolist() for k, v in voice_map.items()}
        output_filename = "voices.json"
        
        try:
            with open(output_filename, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"✅ Successfully exported all voice data to '{output_filename}'.")
        except Exception as e:
            print(f"Error exporting all voices: {e}", file=sys.stderr)
            
    elif voice_name in voice_map:
        # Extract a single voice
        
        # 1. Prepare 2D Array
        voice_array_2d = get_clean_2d_array(voice_map[voice_name])
        
        # 2. Export to JSON (Strict Key:Value Format)
        print(f"Extracting single voice '{voice_name}' (JSON and Binary)...")
        single_voice_data = { voice_name: voice_array_2d.tolist() }
        json_output_filename = f"{voice_name}.json"
        
        try:
            with open(json_output_filename, "w") as f:
                json.dump(single_voice_data, f, indent=4)
            print(f"✅ Successfully exported '{voice_name}' to '{json_output_filename}' (JSON).")
        except Exception as e:
            print(f"Error exporting single voice '{voice_name}' to JSON: {e}", file=sys.stderr)
            
        # 3. Export to Raw Binary (.dat)
        save_voice_to_raw_binary(voice_array_2d, voice_name)

    else:
        print(f"Error: Voice name '{voice_name}' not found in the voice file.", file=sys.stderr)
        
# --- (Other functions like filter_voices, list_voices, short_list_voices, 
# --- show_details, output_vector_as_cpp_pure, and main are unchanged in their logic) ---
# (I will provide the full, runnable main function for completeness)

# --- Remaining functions for completeness ---

def filter_voices(voice_map: Dict[str, np.ndarray], gender_filter: str) -> Dict[str, np.ndarray]:
    target_gender = None
    if gender_filter.lower() == 'm':
        target_gender = 'Male'
    elif gender_filter.lower() == 'f':
        target_gender = 'Female'
    else:
        return voice_map
    
    filtered_map = {}
    for name, data in voice_map.items():
        if get_gender_from_prefix(name) == target_gender:
            filtered_map[name] = data
            
    return filtered_map

def list_voices(voice_map: Dict[str, np.ndarray], gender_filter: str):
    filtered_map = filter_voices(voice_map, gender_filter)
    
    if not filtered_map:
        print(f"No voices found for filter: {gender_filter.upper()}")
        return

    print(f"\n--- Available Voices ({len(filtered_map)} Total) ---")
    
    for i, name in enumerate(sorted(filtered_map.keys())):
        data = filtered_map[name]
        cleaned_data = get_clean_2d_array(data)
        shape_str = str(cleaned_data.shape).replace(" ", "")
        gender_full = get_gender_from_prefix(name)
        print(f"  {i+1:02}: {name:<15} (Shape: {shape_str}, Gender: {gender_full})")
    print("-----------------------------------")


def short_list_voices(voice_map: Dict[str, np.ndarray]):
    if not voice_map:
        return

    for name in sorted(voice_map.keys()):
        print(name)

def show_details(voice_map: Dict[str, np.ndarray], voice_name: str, file_disk_size_mb: Optional[float] = None):
    if not voice_name:
        total_voices = len(voice_map)
        if total_voices == 0:
            print("No voices found.")
            return

        first_voice_data = next(iter(voice_map.values()))
        cleaned_data = get_clean_2d_array(first_voice_data)
        voice_shape = cleaned_data.shape
        voice_dtype = str(cleaned_data.dtype)
        
        bytes_per_float = cleaned_data.itemsize 
        total_data_bytes = sum(get_clean_2d_array(data).size * bytes_per_float for data in voice_map.values())
        total_data_mb = total_data_bytes / (1024 * 1024)
        
        print("\nVoice File Summary")
        if file_disk_size_mb is not None:
            print(f"\nFile Size: {file_disk_size_mb:.2f} MB")
            print("---")
            
        print(f"Voices in file: {total_voices}")
        print(f"Data Payload Size: {total_data_mb:.2f} MB (Vector data only)")
        print(f"Individual Voice Shape: {voice_shape} ({voice_shape[0]} vectors x {voice_shape[1]} dimensions)")
        print(f"Data Type: {voice_dtype}")
        print("--------------------------")
        
    elif voice_name in voice_map:
        data = get_clean_2d_array(voice_map[voice_name])
        gender_full = get_gender_from_prefix(voice_name)
        
        name_parts = voice_name.split('_', 1)
        plain_name = name_parts[1] if len(name_parts) > 1 else "N/A"

        bytes_per_float = data.itemsize 
        single_voice_data_bytes = data.size * bytes_per_float
        single_voice_data_mb = single_voice_data_bytes / (1024 * 1024)
        
        print(f"\nVoice Details: {voice_name}")
        print(f"Name: {plain_name}")
        print(f"Gender: {gender_full}")
        print(f"Shape: {data.shape}")
        print(f"Total Elements: {data.size}")
        print(f"Data Payload Size: {single_voice_data_mb:.2f} MB")
        print(f"Data Type: {data.dtype}")
        print("-----------------------------")
    else:
        print(f"Error: Voice name '{voice_name}' not found.", file=sys.stderr)


def output_vector_as_cpp_pure(voice_array: np.ndarray, voice_name: str):
    vectors = get_clean_2d_array(voice_array)

    if vectors.ndim != 2 or vectors.size == 0:
        print(f"Error: Voice '{voice_name}' data is not a valid 2D array for C++ output.", file=sys.stderr)
        sys.exit(1)

    var_name = re.sub(r'[^a-zA-Z0-9_]', '_', voice_name)
    var_name = var_name.upper()

    print(f"const std::vector<std::vector<float>> {var_name} = {{")

    np.set_printoptions(precision=8, suppress=True, linewidth=1000000) 

    rows = vectors.shape[0]
    
    for i in range(rows):
        row_str = np.array2string(vectors[i], separator=', ', prefix='', suffix='')
        
        row_str = row_str.strip().strip('[]')
        
        print(f"    {{ {row_str} }}", end="")
        
        if i < rows - 1:
            print(",")
        else:
            print("")

    print("};")

def main():
    parser = argparse.ArgumentParser(
        description="Utility to inspect, list, and extract style vectors from the Kokoro TTS voice file (supports .bin/.npy/.npz and .json formats).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'input_file',
        nargs='?',
        default=DEFAULT_VOICE_FILE,
        help=f"Path to the voice data file (.bin, .npy, .npz, or .json).\nDefaults to: '{DEFAULT_VOICE_FILE}'."
    )
    
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        '-l', '--list',
        nargs='?',
        const='all',
        choices=['all', 'm', 'f'], 
        help="List available voices with details. Use 'm' for male or 'f' for female. Default is 'all'.\nExample: -l f"
    )
    
    group.add_argument(
        '-sl', '--short-list',
        action='store_true',
        help="Print only the voice names, one per line, for programmatic parsing."
    )

    group.add_argument(
        '-e', '--extract',
        nargs='?',
        const='all',
        help="Extract voices to JSON. Use a voice name (e.g., 'af_bella') to extract a single voice to <name>.json AND <name>.dat (Raw Binary).\nNo argument (or 'all') extracts all voices to 'voices.json'. Skips 'all' extraction if input is already JSON."
    )
    
    group.add_argument(
        '-v', '--vector',
        help="Extracts a single voice's vectors and outputs them as a C++ std::vector<std::vector<float>> initializer list to stdout (PURE DATA ONLY).\nArgument: <voice_name> (e.g., 'af_bella')"
    )

    group.add_argument(
        '-d', '--detail',
        nargs='?',
        const='summary',
        help="Show file details. Use a voice name (e.g., 'af_bella') for single voice details.\nNo argument ('summary') shows the total voice count, size, and shape."
    )
    
    args = parser.parse_args()
    
    is_pure_output = args.vector is not None

    input_file_path = args.input_file
    
    voice_map = load_voice_data(input_file_path, suppress_output=is_pure_output)
    
    if voice_map is None:
        if is_pure_output:
            sys.exit(1)
        return
    
    # --- Get actual file size on disk for detail command ---
    file_disk_size_mb = None
    if args.detail:
        try:
            file_disk_size_bytes = os.path.getsize(input_file_path)
            file_disk_size_mb = file_disk_size_bytes / (1024 * 1024)
        except OSError:
            file_disk_size_mb = None
    # ----------------------------------------

    # 2. Execute Action
    if args.list:
        list_voices(voice_map, args.list)
        
    elif args.short_list:
        short_list_voices(voice_map)

    elif args.extract:
        voice_name_to_extract = None if args.extract == 'all' else args.extract
        extract_voices(voice_map, voice_name_to_extract, input_file_path.lower().endswith('.json'))
        
    elif args.vector:
        voice_name = args.vector
        if voice_name in voice_map:
            voice_array = voice_map[voice_name]
            output_vector_as_cpp_pure(voice_array, voice_name)
        else:
            print(f"Error: Voice name '{voice_name}' not found in the voice file.", file=sys.stderr)
            sys.exit(1)

    elif args.detail:
        voice_name_for_detail = None if args.detail == 'summary' else args.detail
        disk_size_to_pass = file_disk_size_mb if voice_name_for_detail is None else None
        
        show_details(voice_map, voice_name_for_detail, disk_size_to_pass)

    else:
        # Default behavior when no command is specified (Summary + Help)
        print(f"Input file processed: '{input_file_path}'.")
        print("Run with -h or --help for usage instructions.")
        show_details(voice_map, None, file_disk_size_mb)


if __name__ == "__main__":
    main()
