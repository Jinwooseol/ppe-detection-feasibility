import os

# Never let Ultralytics pip-install missing packages on its own, and keep runs offline-friendly.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_VERBOSE", "True")


def quiet_ultralytics() -> None:
    """Turn off anonymous usage analytics (settings are per user, so set them on every run)."""
    from ultralytics import settings
    if settings.get("sync", True):
        settings.update({"sync": False})
