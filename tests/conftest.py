def pytest_xdist_auto_num_workers(config):
    """Workers for `-n auto` (pyproject.toml): 4 for the fast suite, about 4x faster; none for the slow suite,
    whose module fixtures (e.g. a take tracked once) would run again in every worker."""
    return 4 if config.option.markexpr == "not slow" else 0
