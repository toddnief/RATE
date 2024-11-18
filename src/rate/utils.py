import argparse
import grp
import inspect
import json
import os
import stat

from constants import GROUP_NAME, SCORED_DIR


def set_group_and_permissions(save_path, group_name=GROUP_NAME):
    if group_name is None:
        return
    gid = grp.getgrnam(group_name).gr_gid
    os.chown(save_path, os.getuid(), gid)
    os.chmod(save_path, stat.S_IRUSR | stat.S_IRGRP)


def write_to_json(data, filepath, format="json"):
    if format == "json":
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
    elif format == "jsonl":
        with open(filepath, "w") as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + "\n")
    set_group_and_permissions(filepath)


def load_dataset_from_json(filepath):
    if filepath.suffix == ".jsonl":
        with filepath.open("r") as file:
            dataset = {idx: json.loads(line.strip()) for idx, line in enumerate(file)}
    elif filepath.suffix == ".json":
        with filepath.open("r") as file:
            dataset = json.load(file)
    else:
        raise ValueError(
            "Unsupported file format. Only .json and .jsonl are supported."
        )
    return dataset


def serialize_experiment_template(experiment_template):
    serialized_template = {}

    for key, value in experiment_template.items():
        if callable(value):
            try:
                serialized_template[key] = inspect.getsource(value).strip()
            except TypeError:
                # Fallback to the memory address if we can't get the source
                serialized_template[key] = (
                    f"<unserializable function at {hex(id(value))}>"
                )
        else:
            try:
                if key == "original_completions":
                    serialized_template[key] = "Not serializing the whole dataset :)"
                    continue
                json.dumps(value)  # Attempt to serialize
                serialized_template[key] = value  # If serializable, keep it
            except (TypeError, OverflowError):
                serialized_template[key] = (
                    f"<non-serializable object at {hex(id(value))}>"
                )

    return serialized_template


def combine_jsonl_files(file_paths, output_file, batch_size=1000, file_dir=SCORED_DIR):
    """
    Combines corresponding JSON objects from an arbitrary number of JSONL files in batches.

    Args:
        file_paths: List of paths to the input JSONL files.
        output_file: Path to the output JSONL file.
        batch_size: Number of lines to process in each batch.
    """
    file_handles = [open(file_dir / file_path, "r") for file_path in file_paths]

    output_filepath = file_dir / output_file
    with open(output_filepath, "w") as out:
        while True:
            batch_lines = [list(next_n_lines(f, batch_size)) for f in file_handles]

            # Check if all files are exhausted
            if any(len(lines) == 0 for lines in batch_lines):
                break

            for line_set in zip(*batch_lines):
                json_objects = [json.loads(line) for line in line_set]

                # Note: later files overwrite earlier files in case of key conflicts
                combined_json = {}
                for json_obj in json_objects:
                    combined_json.update(json_obj)

                out.write(json.dumps(combined_json) + "\n")

    set_group_and_permissions(output_filepath)

    for f in file_handles:
        f.close()


def next_n_lines(file_handle, n):
    """
    Generator to yield the next 'n' lines from the file handle.
    """
    for _ in range(n):
        line = file_handle.readline()
        if not line:
            break
        yield line


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Combine multiple JSONL files.")
    parser.add_argument(
        "files",
        nargs="+",  # Accepts one or more file paths
        help="Paths to the JSONL files to combine",
    )
    parser.add_argument(
        "--output", required=True, help="Output file to store the combined JSONL data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of lines to process in each batch (default: 1000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combine_jsonl_files(args.files, args.output, batch_size=args.batch_size)
