from backend.catalog import manifest_errors, selected_index_assets, unregistered_raw_assets


def test_manifest_registers_every_supported_raw_asset():
    assert unregistered_raw_assets() == []
    assert manifest_errors() == []


def test_only_current_criminal_procedure_version_is_selected():
    criminal_procedure = [
        asset for asset in selected_index_assets()
        if asset.title == "형사소송법"
    ]

    assert [asset.version_label for asset in criminal_procedure] == ["법률 제21232호"]
