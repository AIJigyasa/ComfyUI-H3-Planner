"""HTTP routes so the timeline UI works without running the graph.

The card strip needs live state — which segments have clips, which failed,
what the thumbnails look like — and it needs it while you are still editing,
not only after a queue run. Same approach clip_vault.js already uses.

Thumbnails are served by ComfyUI's own /view endpoint; only the timeline JSON
and a couple of mutations need routes of their own.
"""

import os

from . import paths, refine, store, vault


def _project_names():
    root = os.path.join(paths.output_root(), paths.ROOT_SUBFOLDER)
    try:
        return sorted(
            name for name in os.listdir(root)
            if os.path.isfile(os.path.join(root, name, "timeline.json")))
    except OSError:
        return []


def _decorate(project, timeline):
    """Attach clip info to each segment so the UI can draw a card.

    Read from the segment's own ``clips`` record, never by asking the vault
    what it happens to hold. The vault keeps files around after a re-plan, so
    querying it showed a stale clip on a segment the timeline considers
    unrendered — the card looked done when it was pending, and re-planning
    never cleared it.
    """
    for seg in timeline.get("segments", []):
        cards = {}
        for pass_name in ("draft", "final"):
            entry = (seg.get("clips") or {}).get(pass_name)
            if not entry or not entry.get("file"):
                continue
            if not os.path.exists(
                    os.path.join(paths.vault_dir(project), entry["file"])):
                continue
            def served(name):
                if not name:
                    return None
                full = os.path.join(paths.vault_dir(project), name)
                if not os.path.exists(full):
                    return None
                sub, fn = paths.relative_to_output(full)
                # A re-render at the same take reuses the filename, so the
                # browser would serve the previous clip straight from cache.
                # Stamping the file's mtime on the URL is what makes a fresh
                # render actually appear.
                return {"filename": fn, "subfolder": sub, "type": "output",
                        "v": int(os.path.getmtime(full))}

            thumb = served(entry.get("thumb"))
            clip = served(entry.get("file"))  # so a card can play the mp4
            cards[pass_name] = {
                "clip": clip,
                "take": entry.get("take", 0),
                "frames": entry.get("frames", 0),
                "duration": entry.get("duration", 0.0),
                "width": entry.get("width", 0),
                "height": entry.get("height", 0),
                "has_audio": entry.get("has_audio", False),
                "thumb": thumb,
            }
        seg["_vault"] = cards
    return timeline


def register():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as ex:
        print("[H3Planner] routes not registered: %s" % ex)
        return

    routes = PromptServer.instance.routes

    @routes.get("/h3planner/projects")
    async def list_projects(request):
        return web.json_response({"projects": _project_names()})

    @routes.get("/h3planner/timeline")
    async def get_timeline(request):
        project = paths.safe_name(request.query.get("project", ""), "")
        if not project:
            # Never guess. This used to fall back to the first project on
            # disk, so a card strip that asked before it knew its own name
            # came back decorated with ANOTHER project's clips, thumbnails
            # and states: switch workflows, come back, and an advert showed
            # a perfume shoot's renders under its own prompts.
            return web.json_response({"segments": [], "project": "",
                                      "missing": True,
                                      "reason": "no project named"})
        timeline = store.load(paths.timeline_path(project))
        if timeline is None:
            return web.json_response({"segments": [], "project": project,
                                      "missing": True})
        return web.json_response(_decorate(project, timeline))

    @routes.post("/h3planner/segment")
    async def mutate_segment(request):
        """Runtime-only mutations the UI can make between queue runs.

        Authored fields stay in the node's widget — this only touches state,
        so an edit here can never disagree with the workflow you saved.
        """
        body = await request.json()
        project = paths.safe_name(body.get("project", ""), "")
        seg_id = str(body.get("segment_id", ""))
        action = str(body.get("action", ""))
        pass_name = str(body.get("pass", "draft"))

        path = paths.timeline_path(project)
        timeline = store.load(path)
        if timeline is None:
            return web.json_response({"error": "no timeline for %r" % project},
                                     status=404)

        if action not in ("redo_all", "clear_all"):
            seg = store.find(timeline, seg_id)
            if seg is None:
                return web.json_response({"error": "no segment %r" % seg_id},
                                         status=404)

        if action in ("redo_all", "clear_all"):
            passes = ("draft", "final") if action == "clear_all" else (pass_name,)
            touched = 0
            for target in timeline["segments"]:
                if target.get("state") == store.LOCKED:
                    continue
                if not any(target.get("clips", {}).get(p) for p in passes):
                    continue
                for p in passes:
                    target.setdefault("clips", {}).pop(p, None)
                target["state"] = store.PENDING
                target["last_error"] = ""
                target.pop("chosen", None)
                touched += 1
            store.save(path, timeline)
            print("[H3Planner] %s cleared %d segment(s) in %r"
                  % (action, touched, project))
            return web.json_response(_decorate(project, timeline))

        if action == "redo":
            seg.setdefault("clips", {}).pop(pass_name, None)
            seg["state"] = store.PENDING
            seg["last_error"] = ""
        elif action == "lock":
            seg["state"] = store.LOCKED
        elif action == "unlock":
            seg["state"] = (pass_name if seg.get("clips", {}).get(pass_name)
                            else store.PENDING)
        elif action == "reset_running":
            seg["state"] = store.PENDING
            seg["claimed_at"] = None
        elif action == "refine":
            # The one action here that rewrites an authored field. It is
            # allowed because it edits the prompt the planner already
            # wrote, exactly as typing in the card editor does — and the
            # card mirrors it straight back into the widget.
            try:
                note = refine.refine_segment(
                    timeline, seg, body.get("note", ""))
            except Exception as ex:
                # A refusal only reached the browser as a 400, so nothing
                # about a failed refine appeared in the ComfyUI log and it
                # could not be diagnosed after the fact.
                print("[H3Planner] refine %s refused: %s"
                      % (seg_id, str(ex).replace(chr(10), " ")))
                return web.json_response({"error": str(ex)}, status=400)
            seg.setdefault("clips", {}).pop(pass_name, None)
            seg["state"] = store.PENDING
            seg["last_error"] = ""
            seg.pop("chosen", None)
            store.save(path, timeline)
            print("[H3Planner] %s" % note)
            out = _decorate(project, timeline)
            out["message"] = note
            # The card reads its prompt from the node widget, not from
            # disk, so hand the revision back or it would keep showing
            # the old text until the graph was queued again.
            out["segment_id"] = seg["id"]
            out["prompt"] = seg["prompt"]
            return web.json_response(out)
        else:
            return web.json_response({"error": "unknown action %r" % action},
                                     status=400)

        store.save(path, timeline)
        return web.json_response(_decorate(project, timeline))

    @routes.post("/h3planner/upload")
    async def upload_reference(request):
        """Take a reference file dropped on the Cast Board.

        Saved into ComfyUI's input/h3_planner/, which is already served by
        /view, so a reloaded workflow finds its references with no extra
        plumbing and the node reads them straight off disk.
        """
        import re
        from . import nodes_cast

        reader = await request.multipart()
        field = await reader.next()
        while field is not None and field.name != "file":
            field = await reader.next()
        if field is None:
            return web.json_response({"error": "no file in the request"},
                                     status=400)

        original = os.path.basename(field.filename or "reference")
        stem, ext = os.path.splitext(original)
        kind = nodes_cast.kind_of(original)
        if kind == "other":
            return web.json_response(
                {"error": "unsupported file type %r — images, video or audio"
                          % ext}, status=400)

        stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)[:48] or "reference"
        target_dir = paths.cast_dir()
        name = "%s%s" % (stem, ext.lower())
        n = 1
        while os.path.exists(os.path.join(target_dir, name)):
            name = "%s_%d%s" % (stem, n, ext.lower())
            n += 1

        size = 0
        with open(os.path.join(target_dir, name), "wb") as out:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                out.write(chunk)

        return web.json_response({
            "file": name,
            "kind": kind,
            "bytes": size,
            "subfolder": paths.CAST_SUBFOLDER,
            "type": "input",
        })

    @routes.get("/h3planner/references")
    async def list_references(request):
        from . import nodes_cast
        try:
            names = sorted(os.listdir(paths.cast_dir()))
        except OSError:
            names = []
        return web.json_response({"files": [
            {"file": n, "kind": nodes_cast.kind_of(n)} for n in names
            if nodes_cast.kind_of(n) != "other"]})

    print("[H3Planner] routes registered")
