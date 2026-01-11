def pytest_configure() -> None:
    # `multiprocess` can emit a noisy shutdown traceback on Python 3.12+ on some platforms.
    # This is test-only (doesn't affect library runtime) and keeps CI/local output clean.
    try:
        import multiprocess.resource_tracker as resource_tracker
    except Exception:
        return

    original_del = getattr(resource_tracker.ResourceTracker, "__del__", None)
    if original_del is None:
        return

    def _patched_del(self) -> None:  # type: ignore[no-untyped-def]
        try:
            original_del(self)
        except AttributeError as exc:
            if "_recursion_count" in str(exc):
                return
            raise

    resource_tracker.ResourceTracker.__del__ = _patched_del  # type: ignore[assignment]

