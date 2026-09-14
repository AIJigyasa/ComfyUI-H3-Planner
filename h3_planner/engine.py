"""Soft bridge to ComfyUI-H3-Prompt-Creator.

The Segment Prompter reuses that pack's engine rather than copying it — the
provider stack, the Ollama context handling, the truncated-JSON repair, the
shot-timing enforcement, the label renumbering and the subject de-duplication
are all hard-won and would drift immediately as a second copy.

It is a *soft* dependency: without the other pack installed, every node here
still loads and says plainly what is missing.
"""

import importlib.util
import os
import sys
import time

MODULE_FILE = "h3_prompt_creator.py"
_cache = None  # None = not looked yet, False = looked and absent

MISSING = (
    "H3 Segment Prompter needs ComfyUI-H3-Prompt-Creator installed alongside "
    "this pack (it reuses that pack's prompt engine and providers). Install it "
    "into ComfyUI/custom_nodes/ and restart. The Treatment Splitter, and every "
    "other node here, work without it."
)


def _find_loaded():
    """The module ComfyUI already imported when it loaded the other pack."""
    for module in list(sys.modules.values()):
        try:
            path = getattr(module, "__file__", None)
            if path and os.path.basename(path) == MODULE_FILE:
                if hasattr(module, "FULL_REF_SYSTEM"):
                    return module
        except Exception:
            continue
    return None


def _load_from_disk():
    """Import the sibling pack ourselves if ComfyUI hasn't yet.

    It uses relative imports, so it has to be loaded as a package rather than
    as a lone file.
    """
    # Three levels: engine.py -> h3_planner -> ComfyUI-H3-Planner -> custom_nodes.
    # Two landed on this pack's own folder, so the scan looked for the sibling
    # inside ourselves and never found it. The bug stayed hidden because
    # ComfyUI imports the other pack itself and _find_loaded picks it up; this
    # fallback only matters when load order puts us first.
    custom_nodes = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    try:
        entries = sorted(os.listdir(custom_nodes))
    except OSError:
        return None

    for entry in entries:
        pkg_dir = os.path.join(custom_nodes, entry)
        target = os.path.join(pkg_dir, MODULE_FILE)
        if not os.path.isfile(target):
            continue
        pkg_name = "h3planner_bridge_" + "".join(
            c if c.isalnum() else "_" for c in entry)
        try:
            if pkg_name not in sys.modules:
                init = os.path.join(pkg_dir, "__init__.py")
                spec = importlib.util.spec_from_file_location(
                    pkg_name, init, submodule_search_locations=[pkg_dir])
                module = importlib.util.module_from_spec(spec)
                sys.modules[pkg_name] = module
                spec.loader.exec_module(module)
            creator = importlib.import_module(pkg_name + ".h3_prompt_creator")
            if hasattr(creator, "FULL_REF_SYSTEM"):
                print("[H3Planner] prompt engine loaded from %s" % entry)
                return creator
        except Exception as ex:
            print("[H3Planner] could not load prompt engine from %s: %s"
                  % (entry, ex))
    return None


_last_look = 0.0
_RETRY_AFTER = 20.0


def creator():
    """The sibling pack's module, or None.

    A success is cached forever; a failure is only cached briefly. Caching a
    failure permanently meant one transient miss — the other pack still
    importing, a partially written file during an update — left every node in
    this pack on the Ollama-only provider list for the rest of the session,
    with the cloud providers silently absent and a restart the only cure.
    """
    global _cache, _last_look
    if _cache:
        return _cache
    now = time.time()
    if _cache is False and now - _last_look < _RETRY_AFTER:
        return None
    _last_look = now
    _cache = _find_loaded() or _load_from_disk() or False
    return _cache or None


def available():
    return creator() is not None


def require():
    mod = creator()
    if mod is None:
        raise RuntimeError(MISSING)
    return mod


# -- thin wrappers so callers never touch the other pack's private names ----

def providers():
    mod = creator()
    if mod is None:
        return ["Ollama (Local)"]
    try:
        return list(mod.h3_providers.PROVIDERS)
    except Exception:
        return ["Ollama (Local)"]


def default_ollama_model():
    return "qwen3-vl:8b"


def backend_config(**kwargs):
    return require().BackendConfig(**kwargs)


def json_schema(properties, required):
    return require()._json_schema(properties, required)


def generate(cfg, system, user, schema, required_keys=(), min_words=0,
             images=None):
    """One structured generation. Returns (obj_or_None, note)."""
    return require()._generate_with_provider(
        cfg, system, user, images or [], schema,
        required_keys=required_keys, min_words=min_words)


def image_b64(path):
    """One reference image, encoded the way the provider stack expects.

    Without this the planner writes what a subject looks like from the brief
    alone — it has never seen the photograph. A cast image of one person and a
    written description of a different one is a contradiction, and H3 resolves
    contradictions by blending them.
    """
    mod = creator()
    if mod is None:
        return None
    try:
        from PIL import Image
        with Image.open(path) as img:
            return mod._pil_to_b64(img.convert("RGB"))
    except Exception as ex:
        print("[H3Planner] reference not readable for the planner (%s): %s"
              % (path, ex))
        return None


def render_full_ref(obj, duration=0.0):
    return require()._render_full_ref(obj, duration)


def fix_shot_times(text, duration):
    return require()._fix_shot_times(text, duration)


def normalize_labels(text, duration=0.0):
    return require()._normalize_h3_labels(text, duration)


def dedupe_subjects(text):
    return require()._dedupe_subject_definitions(text)


def clean(value):
    mod = creator()
    if mod is None:
        return "" if value is None else str(value).strip()
    return mod._clean(value)


def full_ref_system():
    return require().FULL_REF_SYSTEM
