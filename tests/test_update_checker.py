from spejl.gui.update_checker import find_update


def test_new_release_returns_installer_asset():
    info = find_update(
        {
            "tag_name": "v0.1.2",
            "html_url": "https://github.com/immanuelserdan-ui/spejl/releases/tag/v0.1.2",
            "assets": [
                {
                    "name": "Spejl-0.1.2-Windows-x64-Setup.exe",
                    "browser_download_url": "https://example/setup.exe",
                }
            ],
        },
        current_version="0.1.1",
    )
    assert info is not None
    assert info.version == "0.1.2"
    assert info.download_url.endswith("setup.exe")


def test_same_release_is_ignored():
    assert find_update({"tag_name": "v0.1.1", "assets": []}, current_version="0.1.1") is None
