import pyreth


def test_version_string():
    assert pyreth.__version__ is not None
    assert isinstance(pyreth.__version__, str)
    assert len(pyreth.__version__) > 0
    # SEMVER: MAJOR.MINOR.PATCH
    assert len(pyreth.__version__.split(".")) == 3


def test_runtime_version():
    runtime_version = pyreth.get_runtime_version()
    assert runtime_version is not None
    assert isinstance(runtime_version, str)
    assert len(runtime_version) > 0


def test_best_backend():
    backend = pyreth.get_best_backend()
    assert backend is not None
    assert isinstance(backend, str)
    assert len(backend) > 0


def test_hardware_acceleration():
    has_hw = pyreth.has_hardware_acceleration()
    assert isinstance(has_hw, bool)
