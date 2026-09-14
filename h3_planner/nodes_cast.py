"""H3 Cast Board — the reference cast, uploaded into the node itself.

Roles decide tags, not file order. The H3 guide is explicit that a character,
product or style image is NOT automatically a <Picture N>: if it only supplies
reusable identity it is a <Subject N>, and <Picture N> is reserved for a
concrete first/last/key/edited frame or composition anchor. The four kinds are
numbered independently, so <Subject 1> and <Picture 1> can be different assets.

Numbering follows card order within each kind, so reordering the board renumbers
the tags — and the wired IMAGE outputs move with them, which is what keeps the
tags in a prompt pointing at the slot they were written for.

The three reference inputs grow on their own: connect ref_image_0 and
ref_image_1 appears. That is ComfyUI's own Autogrow, the same mechanism the
MiniMax H3 sampler uses for its ref_image_ / ref_video_ / ref_audio_ families,
so the two nodes behave identically. Outputs cannot do the same: an output link
is bound to a slot INDEX, so a node that adds or hides output slots silently
repoints every connection downstream of it. They are a fixed set instead.
"""

import json
import os

from . import media, paths

CATEGORY = "H3 Planner"

# Output slots, fixed. Inputs grow; these cannot — see the module docstring.
MAX_IMAGE_SLOTS = 8
MAX_AUDIO_SLOTS = 4
MAX_VIDEO_SLOTS = 4

# The ceiling on each autogrow family, matching the H3 sampler's own limits so
# a full board can be wired straight across without running out of sockets.
MAX_IMAGE_INPUTS = 9
MAX_AUDIO_INPUTS = 4
MAX_VIDEO_INPUTS = 4

MAX_SLOTS = MAX_IMAGE_SLOTS     # the old name, kept for anything importing it

# role -> H3 tag kind
ROLE_TAGS = {
    "character": "Subject",
    "product": "Subject",
    "style": "Subject",
    "wardrobe": "Subject",
    "environment": "Subject",
    "prop": "Subject",
    "first_frame": "Picture",
    "last_frame": "Picture",
    "keyframe": "Picture",
    "composition": "Picture",
    "video": "Video",
    "audio": "Audio",
}

ROLES = list(ROLE_TAGS.keys())

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def kind_of(filename):
    ext = os.path.splitext(str(filename))[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return "other"


def load_image(path):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def wired_values(grown):
    """An Autogrow bundle as a plain list, in slot order.

    Autogrow hands back ``{"ref_image_0": tensor, "ref_image_3": tensor}`` with
    no guarantee about dict order and with gaps where a socket was left empty.
    Sorting by the numeric suffix is what makes the tag numbering follow what
    the node looks like rather than what order the frontend happened to send.
    """
    if not grown:
        return []
    if not isinstance(grown, dict):
        grown = {str(i): v for i, v in enumerate(grown)}

    def index(name):
        digits = "".join(c for c in str(name).rsplit("_", 1)[-1] if c.isdigit())
        return int(digits) if digits else 0

    return [grown[k] for k in sorted(grown, key=index) if grown[k] is not None]


def trim_of(entry):
    """(start, end) seconds for one card, or (0.0, None) for the whole file.

    Only audio and video carry one. A blank field, a zero, or a missing key all
    mean "no trim", so the UI can leave the inputs empty.
    """
    def number(key):
        raw = entry.get(key)
        if raw in (None, "", False):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError("trim %s %r is not a number" % (key, raw))
        return value if value > 0 else None

    start, end = number("start"), number("end")
    if start and end and end <= start:
        raise ValueError("trim ends at %.3fs but starts at %.3fs" % (end, start))
    return (start or 0.0), end


def assemble(cast_json, wired_images=(), wired_audios=(), wired_videos=()):
    """The whole board: cards plus wired references, tagged and numbered.

    Returns ``(cast, images, audios, videos, report)`` with the three lists
    already padded to their output slot counts.
    """
    try:
        doc = json.loads(cast_json or "{}")
    except json.JSONDecodeError as ex:
        raise ValueError("cast_json is not valid JSON: %s" % ex.msg)
    entries = doc.get("entries") or []

    root = paths.cast_dir()
    members, notes = [], []
    images, audios, videos, video_paths = [], [], [], []
    counters = {"Subject": 0, "Picture": 0, "Video": 0, "Audio": 0}

    def full(bucket, limit, what, key):
        """True when this kind has no slot left, and says so honestly.

        The overflow is dropped from the cast entirely rather than tagged.
        A tag that reaches a prompt with no asset behind it renders as
        nothing at all, which is a worse failure than a card that visibly
        did not make it.
        """
        if len(bucket) < limit:
            return False
        notes.append(
            "%s: past the %d %s slot(s) — left out of the cast, so no prompt "
            "can cite it. Disable a card of the same kind to make room."
            % (key, limit, what))
        return True

    for entry in entries:
        if entry.get("disabled"):
            continue
        role = entry.get("role") or "character"
        if role not in ROLE_TAGS:
            notes.append("unknown role %r on %r — treated as character"
                         % (role, entry.get("key")))
            role = "character"
        kind = ROLE_TAGS[role]
        filename = entry.get("file") or ""
        path = os.path.join(root, filename) if filename else ""
        key = entry.get("key") or role

        if filename and not os.path.exists(path):
            notes.append("%s: file %s is missing from input/%s"
                         % (key, filename, paths.CAST_SUBFOLDER))
            continue

        try:
            start, end = trim_of(entry)
        except ValueError as ex:
            notes.append("%s: %s — trim ignored" % (key, ex))
            start, end = 0.0, None

        asset_kind = kind_of(filename)
        if kind in ("Subject", "Picture"):
            if asset_kind != "image":
                notes.append("%s: role %s needs an image, got %s"
                             % (key, role, filename))
                continue
            if full(images, MAX_IMAGE_SLOTS, "image", key):
                continue
            try:
                images.append(load_image(path))
            except Exception as ex:
                notes.append("%s: could not read %s (%s)"
                             % (key, filename, ex))
                continue
        elif kind == "Audio":
            if full(audios, MAX_AUDIO_SLOTS, "audio", key):
                continue
            try:
                audios.append(media.load_audio_file(path, start=start,
                                                    end=end))
            except Exception as ex:
                notes.append("%s: could not decode %s (%s)"
                             % (key, filename, ex))
                continue
        elif kind == "Video":
            if full(videos, MAX_VIDEO_SLOTS, "video", key):
                continue
            try:
                videos.append(media.decode_video(path, start=start,
                                                 end=end))
                video_paths.append(path)
            except Exception as ex:
                # The path still reaches the prompt bridge, so a video that
                # will not decode is a missing output slot, not a dead cast.
                notes.append("%s: %s tagged, but its frames could not be "
                             "decoded (%s)" % (key, filename, ex))
                videos.append(None)
                video_paths.append(path)

        counters[kind] += 1
        members.append({
            "slot": len(images) if kind in ("Subject", "Picture") else 0,
            "key": entry.get("key") or "%s_%d" % (role, counters[kind]),
            "role": role,
            "note": entry.get("note", ""),
            "kind": kind,
            "number": counters[kind],
            "tag": "<%s %d>" % (kind, counters[kind]),
            "file": filename,
            "trim": [start, end] if (start or end) else None,
        })

    # Anything wired in lands after the uploads, numbered on from them.
    for i, img in enumerate(wired_images, start=1):
        if full(images, MAX_IMAGE_SLOTS, "image", "wired ref_image %d" % i):
            continue
        images.append(img)
        counters["Subject"] += 1
        members.append({
            "slot": len(images), "key": "wired_image_%d" % i,
            "role": "character",
            "note": "connected to a ref_image input",
            "kind": "Subject", "number": counters["Subject"],
            "tag": "<Subject %d>" % counters["Subject"], "file": "",
        })

    for i, aud in enumerate(wired_audios, start=1):
        if full(audios, MAX_AUDIO_SLOTS, "audio", "wired ref_audio %d" % i):
            continue
        audios.append(aud)
        counters["Audio"] += 1
        members.append({
            "slot": 0, "key": "wired_audio_%d" % i, "role": "audio",
            "note": "connected to a ref_audio input",
            "kind": "Audio", "number": counters["Audio"],
            "tag": "<Audio %d>" % counters["Audio"], "file": "",
        })

    for i, vid in enumerate(wired_videos, start=1):
        if full(videos, MAX_VIDEO_SLOTS, "video", "wired ref_video %d" % i):
            continue
        videos.append(vid)
        counters["Video"] += 1
        members.append({
            "slot": 0, "key": "wired_video_%d" % i, "role": "video",
            "note": "connected to a ref_video input",
            "kind": "Video", "number": counters["Video"],
            "tag": "<Video %d>" % counters["Video"], "file": "",
        })

    def pad(bucket, size):
        return list(bucket) + [None] * (size - len(bucket))

    audio_tag = next((m["tag"] for m in members if m["kind"] == "Audio"), "")

    cast = {
        "members": members,
        # Image slots only: every caller of `size` is checking how far a
        # <Subject N> or <Picture N> citation may go.
        "size": len(images),
        "counts": dict(counters),
        "audio_tag": audio_tag,
        "has_audio": any(a is not None for a in audios),
        "has_video": bool(video_paths),
        "video_paths": video_paths,
    }

    def described(m):
        bits = [m["role"]]
        if m.get("trim"):
            lo, hi = m["trim"]
            bits.append("trimmed %.2fs to %s"
                        % (lo, "%.2fs" % hi if hi else "the end"))
        return ", ".join(bits)

    tag_reference = "\n".join(
        "%s = %s (%s)%s" % (m["tag"], m["key"], described(m),
                            ("; " + m["note"]) if m.get("note") else "")
        for m in members) or "(no references yet — drop images on the node)"

    report = "\n".join([
        "%d reference(s): %s"
        % (len(members), ", ".join("%d %s" % (v, k)
                                   for k, v in counters.items() if v) or "none"),
        "wired out: %d image, %d audio, %d video slot(s)"
        % (len(images), len(audios), len([v for v in videos if v is not None])),
        ("video files: %s" % ", ".join(os.path.basename(p) for p in video_paths))
        if video_paths else "",
        "",
        tag_reference,
    ] + (["", "NOTES", "! " + "\n! ".join(notes)] if notes else []))

    return (cast, pad(images, MAX_IMAGE_SLOTS), pad(audios, MAX_AUDIO_SLOTS),
            pad(videos, MAX_VIDEO_SLOTS), report)


OUTPUT_NAMES = (("cast",)
                + tuple("image_%d" % i for i in range(1, MAX_IMAGE_SLOTS + 1))
                + tuple("audio_%d" % i for i in range(1, MAX_AUDIO_SLOTS + 1))
                + tuple("video_%d" % i for i in range(1, MAX_VIDEO_SLOTS + 1))
                + ("report",))

CAST_JSON_TOOLTIP = ("managed by the Cast Board UI; press JSON on the node to "
                     "edit it by hand")

IMAGE_TIP = "a reference image. Connect one and the next socket appears."
AUDIO_TIP = "a reference track. Connect one and the next socket appears."
VIDEO_TIP = ("reference video FRAMES, as the H3 sampler wants them. Connect a "
             "video loader's IMAGE output, not a VIDEO object.")


def _flatten(cast, images, audios, videos, report):
    return (cast,) + tuple(images) + tuple(audios) + tuple(videos) + (report,)


try:
    from comfy_api.latest import io as comfy_io
except Exception:      # standalone, e.g. the test suite
    comfy_io = None


if comfy_io is not None:

    class H3PlannerCastBoard(comfy_io.ComfyNode):
        """Upload references here; the node assigns and numbers the H3 tags."""

        @classmethod
        def define_schema(cls):
            return comfy_io.Schema(
                node_id="H3PlannerCastBoard",
                display_name="H3 Cast Board",
                category=CATEGORY,
                description=__doc__,
                inputs=[
                    comfy_io.String.Input("cast_json", multiline=True,
                                          default=json.dumps({"entries": []}),
                                          tooltip=CAST_JSON_TOOLTIP),
                    comfy_io.Autogrow.Input(
                        "ref_images", optional=True,
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Image.Input("ref_image",
                                                       tooltip=IMAGE_TIP),
                            # min=0, not 1. Autogrow files the first `min`
                            # sockets under "required", so min=1 meant a board
                            # that only ever uses uploads refused to run with
                            # "Required input slots have no connection feeding
                            # them". The frontend still draws one spare socket.
                            prefix="ref_image_", min=0,
                            max=MAX_IMAGE_INPUTS)),
                    comfy_io.Autogrow.Input(
                        "ref_audios", optional=True,
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Audio.Input("ref_audio",
                                                       tooltip=AUDIO_TIP),
                            prefix="ref_audio_", min=0,
                            max=MAX_AUDIO_INPUTS)),
                    comfy_io.Autogrow.Input(
                        "ref_videos", optional=True,
                        template=comfy_io.Autogrow.TemplatePrefix(
                            input=comfy_io.Image.Input("ref_video",
                                                       tooltip=VIDEO_TIP),
                            prefix="ref_video_", min=0,
                            max=MAX_VIDEO_INPUTS)),
                ],
                outputs=[comfy_io.Custom("H3_CAST").Output(display_name="cast")]
                + [comfy_io.Image.Output(display_name="image_%d" % i)
                   for i in range(1, MAX_IMAGE_SLOTS + 1)]
                + [comfy_io.Audio.Output(display_name="audio_%d" % i)
                   for i in range(1, MAX_AUDIO_SLOTS + 1)]
                + [comfy_io.Image.Output(display_name="video_%d" % i)
                   for i in range(1, MAX_VIDEO_SLOTS + 1)]
                + [comfy_io.String.Output(display_name="report")],
            )

        @classmethod
        def fingerprint_inputs(cls, cast_json, **kwargs):
            # Uploaded files can change under a stable card list.
            return cast_json

        @classmethod
        def execute(cls, cast_json, ref_images=None, ref_audios=None,
                    ref_videos=None):
            parts = assemble(cast_json,
                             wired_values(ref_images),
                             wired_values(ref_audios),
                             wired_values(ref_videos))
            return comfy_io.NodeOutput(*_flatten(*parts))

        # The test suite and anything older call build() directly. Same code
        # path as execute, so what the tests exercise is what the node runs.
        def build(self, cast_json, ref_images=None, ref_audios=None,
                  ref_videos=None):
            return _flatten(*assemble(cast_json,
                                      wired_values(ref_images),
                                      wired_values(ref_audios),
                                      wired_values(ref_videos)))

else:

    class H3PlannerCastBoard:
        """Classic-schema stand-in for environments without comfy_api.

        Only the tests reach this. The autogrow inputs are flattened into
        plain optional sockets so the shape can still be checked; the
        assembling below is the same function the real node calls.
        """

        @classmethod
        def INPUT_TYPES(cls):
            optional = {"ref_image_%d" % i: ("IMAGE",)
                        for i in range(MAX_IMAGE_INPUTS)}
            optional.update({"ref_audio_%d" % i: ("AUDIO",)
                             for i in range(MAX_AUDIO_INPUTS)})
            optional.update({"ref_video_%d" % i: ("IMAGE",)
                             for i in range(MAX_VIDEO_INPUTS)})
            return {
                "required": {
                    "cast_json": ("STRING", {
                        "multiline": True,
                        "default": json.dumps({"entries": []}),
                        "tooltip": CAST_JSON_TOOLTIP}),
                },
                "optional": optional,
            }

        RETURN_TYPES = (("H3_CAST",) + ("IMAGE",) * MAX_IMAGE_SLOTS
                        + ("AUDIO",) * MAX_AUDIO_SLOTS
                        + ("IMAGE",) * MAX_VIDEO_SLOTS + ("STRING",))
        RETURN_NAMES = OUTPUT_NAMES
        FUNCTION = "build"
        CATEGORY = CATEGORY

        @classmethod
        def IS_CHANGED(cls, cast_json, **kwargs):
            return cast_json

        def build(self, cast_json, ref_images=None, ref_audios=None,
                  ref_videos=None, **wired):
            def bundle(prefix):
                return {k: v for k, v in wired.items() if k.startswith(prefix)}
            return _flatten(*assemble(
                cast_json,
                wired_values(ref_images or bundle("ref_image_")),
                wired_values(ref_audios or bundle("ref_audio_")),
                wired_values(ref_videos or bundle("ref_video_"))))


NODE_CLASS_MAPPINGS = {"H3PlannerCastBoard": H3PlannerCastBoard}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PlannerCastBoard": "H3 Cast Board"}
