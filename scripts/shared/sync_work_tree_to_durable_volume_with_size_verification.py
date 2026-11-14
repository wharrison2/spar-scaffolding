"""One-way synchronisation of a work tree on fast local disk to a durable folder (a RunPod network volume), and the reverse restore. Standard
library only (rsync is not guaranteed to exist in the pod image).

Why: a network volume is slow for the many small appends that answer and judge files get, so the overnight chain's steps write on the
container disk and this copies the changed files to the volume every few minutes and after every step. At most the last interval's work can be
lost with the pod, and every script of the chain resumes from its files, so it regenerates that work.

sync_tree(source, destination, exclude_patterns): copies every file that is new or whose size or modification time differs from the copy
(rsync-style quick check; the copy gets the source's modification time, so an unchanged file is skipped next time). Each file is copied to
<name>.syncing_partial then renamed (a reader never sees half a file), and its size is verified against the bytes read; a file that was
growing is picked up by the next sync because the source's modification time then differs. Returns a dictionary with counts, bytes and
`problems` (empty when every copy verified). Files that vanish during the walk are skipped. Exclude patterns are fnmatch patterns on the
file's path relative to the source, with forward slashes (for example "*/lora_adapters/*").

restore_tree(volume_folder, work_folder, exclude_patterns): the same copy in the other direction, used on a relaunch to rebuild the work tree
from the volume before the chain starts; files already in the work tree that are the same size and not older are kept.
"""

import fnmatch
import os
from pathlib import Path

PARTIAL_SUFFIX = ".syncing_partial"
COPY_CHUNK_BYTES = 1 << 24


def is_excluded(relative_posix_path, exclude_patterns):
    return any(fnmatch.fnmatch(relative_posix_path, pattern) for pattern in exclude_patterns)


def copy_one_file_verified(source_path, destination_path):
    """Returns the number of bytes copied. Raises OSError when the copy's size is not what was read."""
    source_mtime_ns = os.stat(source_path).st_mtime_ns
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial = destination_path.with_name(destination_path.name + PARTIAL_SUFFIX)
    copied = 0
    with open(source_path, "rb") as reader, open(partial, "wb") as writer:
        for chunk in iter(lambda: reader.read(COPY_CHUNK_BYTES), b""):
            writer.write(chunk)
            copied += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    os.utime(partial, ns=(source_mtime_ns, source_mtime_ns))
    if os.stat(partial).st_size != copied:
        raise OSError(f"copy of {source_path} has {os.stat(partial).st_size} bytes, expected {copied}")
    os.replace(partial, destination_path)
    return copied


def file_needs_copy(source_stat, destination_path):
    try:
        destination_stat = os.stat(destination_path)
    except FileNotFoundError:
        return True
    return destination_stat.st_size != source_stat.st_size or destination_stat.st_mtime_ns != source_stat.st_mtime_ns


def sync_tree(source, destination, exclude_patterns=(), keep_newer_destination=False):
    source, destination = Path(source), Path(destination)
    result = {"files_copied": 0, "files_unchanged": 0, "bytes_copied": 0, "problems": []}
    if not source.is_dir():
        return result
    for root, directory_names, file_names in os.walk(source):
        for file_name in file_names:
            if file_name.endswith(PARTIAL_SUFFIX):
                continue
            path = Path(root) / file_name
            relative = path.relative_to(source).as_posix()
            if is_excluded(relative, exclude_patterns):
                continue
            try:
                source_stat = os.stat(path)
                target = destination / relative
                if keep_newer_destination and target.exists() and os.stat(target).st_size == source_stat.st_size \
                        and os.stat(target).st_mtime_ns >= source_stat.st_mtime_ns:
                    result["files_unchanged"] += 1
                    continue
                if not file_needs_copy(source_stat, target):
                    result["files_unchanged"] += 1
                    continue
                result["bytes_copied"] += copy_one_file_verified(path, target)
                result["files_copied"] += 1
            except FileNotFoundError:
                continue  # vanished during the walk
            except OSError as error:
                result["problems"].append(f"{relative}: {error}")
    return result


def restore_tree(volume_folder, work_folder, exclude_patterns=()):
    return sync_tree(volume_folder, work_folder, exclude_patterns, keep_newer_destination=True)


def verify_every_source_file_is_on_the_destination(source, destination, exclude_patterns=()):
    """Problems (missing or different size) after a sync, for files that are not being written at this moment."""
    problems = []
    for root, _, file_names in os.walk(source):
        for file_name in file_names:
            path = Path(root) / file_name
            relative = path.relative_to(source).as_posix()
            if file_name.endswith(PARTIAL_SUFFIX) or is_excluded(relative, exclude_patterns):
                continue
            target = Path(destination) / relative
            try:
                if not target.exists() or target.stat().st_size != path.stat().st_size:
                    problems.append(relative)
            except FileNotFoundError:
                continue
    return problems
