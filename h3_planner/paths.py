"""Where H3 Planner keeps its state.

Everything lives under ComfyUI's output directory so it survives restarts and
is reachable by the frontend:

    output/h3_planner/<project>/timeline.json
    output/h3_planner/<project>/vault/<segment>__<pass>__t<take>.mp4
    output/h3_planner/<project>/vault/vault.json
    output/h3_planner/<project>/frames/<segment>_last.png
    output/h3_planner/<project>/renders/<project>_<stamp>.mp4
"""

import os
import re

ROOT_SUBFOLDER = "h3_planner"
CAST_SUBFOLDER = "h3_planner"  # under ComfyUI's input/ directory

try:  # inside ComfyUI
    import folder_paths

    def output_root():
        return folder_paths.get_output_directory()

    def input_root():
        return folder_paths.get_input_directory()
except Exception:  # standalone (tests)
    def output_root():
        return os.path.join(os.path.expanduser("~"), "h3_planner_output")

    def input_root():
        return os.path.join(os.path.expanduser("~"), "h3_planner_input")


def safe_name(name, fallback="project"):
    name = re.sub(r"[^A-Za-z0-9_\-. ]", "", str(name)).strip()
    return name or fallback


def project_dir(project_name, create=True):
    d = os.path.join(output_root(), ROOT_SUBFOLDER, safe_name(project_name))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _sub(project_name, leaf, create=True):
    d = os.path.join(project_dir(project_name, create), leaf)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def vault_dir(project_name, create=True):
    return _sub(project_name, "vault", create)


def frames_dir(project_name, create=True):
    return _sub(project_name, "frames", create)


def renders_dir(project_name, create=True):
    return _sub(project_name, "renders", create)


def timeline_path(project_name):
    return os.path.join(project_dir(project_name), "timeline.json")


def relative_to_output(path):
    """Path as ComfyUI's /view endpoint wants it: (subfolder, filename)."""
    root = os.path.normpath(output_root())
    full = os.path.normpath(path)
    try:
        rel = os.path.relpath(full, root)
    except ValueError:
        return "", os.path.basename(full)
    sub = os.path.dirname(rel).replace(os.sep, "/")
    return sub, os.path.basename(rel)


def planned_projects():
    """Project names that already have a timeline on disk."""
    root = os.path.join(output_root(), ROOT_SUBFOLDER)
    try:
        return sorted(
            name for name in os.listdir(root)
            if os.path.isfile(os.path.join(root, name, "timeline.json")))
    except OSError:
        return []


def cast_dir(create=True):
    """Where Cast Board uploads live, under ComfyUI's input/ directory.

    Uploads go to input/ rather than output/ because that is where ComfyUI
    already serves user-supplied assets from, so a reloaded workflow finds its
    references without any extra plumbing.
    """
    d = os.path.join(input_root(), CAST_SUBFOLDER)
    if create:
        os.makedirs(d, exist_ok=True)
    return d
