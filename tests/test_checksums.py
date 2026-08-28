from brainon_ai.common.checksums import sha256_file, verify_sha256


def test_sha256(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"brainon")
    assert sha256_file(path) == "dda237ab95460095708e23a476c9f4b8a95ffbd94b761719a726711289a4dd30"
    assert verify_sha256(path, sha256_file(path))
