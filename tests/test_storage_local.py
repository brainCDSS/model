from brainon_ai.common.checksums import sha256_file
from brainon_ai.storage.backends.local import LocalBackend


def test_local_upload_download_partial_safe(tmp_path):
    backend = LocalBackend()
    source = tmp_path / "source.txt"
    source.write_text("data")
    uri = (tmp_path / "remote/value.txt").as_uri()
    backend.upload(source, uri)
    destination = tmp_path / "download/value.txt"
    backend.download(uri, destination, sha256_file(source))
    assert destination.read_text() == "data"
    assert not destination.with_name("value.txt.partial").exists()
