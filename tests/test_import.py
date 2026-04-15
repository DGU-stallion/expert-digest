def test_package_imports_with_version():
    import expert_digest

    assert expert_digest.__version__ == "0.1.0"
