# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================

import functools
import os
import posixpath
import random
import sys
import time
from urllib.parse import urlparse

import pytest
import tensorflow as tf
import tensorflow_io as tfio  # pylint: disable=unused-import

pytest.skip("file system tests is failing with xdist", allow_module_level=True)

# `ROOT_PREFIX` shouldn't be called directly in tests.
ROOT_PREFIX = f"tf-io-root-{int(time.time())}/"

# This is the number of attributes each filesystem should return in `*_fs`.
NUM_ATR_FS = 6

S3_URI = "s3"
AZ_URI = "az"
AZ_DSN_URI = "az_dsn"
HTTPS_URI = "https"
GCS_URI = "gs"


def mock_patchs(monkeypatch, patchs):
    if isinstance(patchs, dict):
        for key, value in patchs.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
    elif callable(patchs):
        patchs(monkeypatch)
    else:
        pass


@pytest.fixture(autouse=True)
def reload_filesystem():
    # We need to find a way to reload `tensorflow` or filesystems since
    # the envs are read in the first time the filesystems are called.
    pass


# Helper to check if we should skip tests for an `uri`.
def should_skip(uri, check_only=True):
    message = None
    if uri == S3_URI and sys.platform in ("win32", "darwin"):
        message = "TODO: `s3` emulator not setup properly on macOS/Windows yet"
    elif uri in (AZ_URI, AZ_DSN_URI) and sys.platform == "win32":
        message = "TODO: `az` does not work on Windows yet"
    elif uri == GCS_URI and sys.platform in ("win32", "darwin"):
        message = "TODO: `gs` does not work on Windows yet"

    if message is not None:
        if check_only:
            return True
        else:
            pytest.skip(message)
    else:
        return False


@pytest.fixture(scope="module")
def s3_fs():
    if should_skip(S3_URI):
        yield [None] * NUM_ATR_FS
        return

    import boto3

    monkeypatch = pytest.MonkeyPatch()
    bucket_name = os.environ.get("S3_TEST_BUCKET")
    client = None

    # This means we are running against emulator.
    if bucket_name is None:
        endpoint_url = "http://localhost:4566"
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "TEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "TEST")
        monkeypatch.setenv("S3_ENDPOINT", endpoint_url)

        bucket_name = f"tf-io-bucket-s3-{int(time.time())}"
        client = boto3.client("s3", endpoint_url=endpoint_url)
        client.create_bucket(Bucket=bucket_name)
    else:
        # TODO(vnvo2409): Implement for testing against production scenario
        pass

    client.put_object(Bucket=bucket_name, Key=ROOT_PREFIX, Body="")

    def parse(path):
        res = urlparse(path, scheme=S3_URI, allow_fragments=False)
        return res.netloc, res.path[1:]

    def path_to(*args):
        return f"{S3_URI}://{bucket_name}/{posixpath.join(ROOT_PREFIX, *args)}"

    def read(path):
        bucket_name, key_name = parse(path)
        response = client.get_object(Bucket=bucket_name, Key=key_name)
        return response["Body"].read()

    def write(path, body):
        bucket_name, key_name = parse(path)
        client.put_object(Bucket=bucket_name, Key=key_name, Body=body)

    def mkdirs(path):
        if path[-1] != "/":
            path += "/"
        write(path, b"")

    yield path_to, read, write, mkdirs, posixpath.join, (client, bucket_name)
    monkeypatch.undo()


@pytest.fixture(scope="module")
def az_fs():
    if should_skip(AZ_URI):
        yield [None] * NUM_ATR_FS
        return

    from azure.storage.blob import ContainerClient

    monkeypatch = pytest.MonkeyPatch()
    container_name = os.environ.get("AZ_TEST_CONTAINER")
    account = None
    client = None

    # This means we are running against emulator.
    if container_name is None:
        monkeypatch.setenv("TF_AZURE_USE_DEV_STORAGE", "1")
        container_name = f"tf-io-bucket-az-{int(time.time())}"
        account = "devstoreaccount1"
        conn_str = (
            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
            "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq"
            "/K1SZFPTOtr/KBHBeksoGMGw==;"
            "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
        )
        client = ContainerClient.from_connection_string(conn_str, container_name)
        client.create_container()
    else:
        # TODO(vnvo2409): Implement for testing against production scenario
        pass

    client.upload_blob(ROOT_PREFIX, b"")

    def parse(path):
        res = urlparse(path, scheme=AZ_URI, allow_fragments=False)
        return res.path.split("/", 2)[2]

    def path_to(*args):
        return f"{AZ_URI}://{account}/{container_name}/{posixpath.join(ROOT_PREFIX, *args)}"

    def read(path):
        key_name = parse(path)
        return client.download_blob(key_name).content_as_bytes()

    def write(path, body):
        key_name = parse(path)
        client.upload_blob(key_name, body)

    def mkdirs(path):
        if path[-1] == "/":
            write(path, b"")

    yield path_to, read, write, mkdirs, posixpath.join, (
        client,
        container_name,
        account,
    )
    monkeypatch.undo()


@pytest.fixture(scope="module")
def az_dsn_fs(az_fs):
    if should_skip(AZ_DSN_URI):
        yield [None] * NUM_ATR_FS
        return

    _, read, write, mkdirs, join, fs_internal = az_fs
    _, container_name, account = fs_internal

    def path_to_dsn(*args):
        return f"{AZ_URI}://{account}.blob.core.windows.net/{container_name}/{posixpath.join(ROOT_PREFIX, *args)}"

    yield path_to_dsn, read, write, mkdirs, join, fs_internal


@pytest.fixture(scope="module")
def https_fs():
    if should_skip(HTTPS_URI):
        yield [None] * NUM_ATR_FS
        return

    def path_to(*_):
        return f"{HTTPS_URI}://www.apache.org/licenses/LICENSE-2.0.txt"

    def read(_):
        pass

    def write(*_):
        pass

    def mkdirs(_):
        pass

    yield path_to, read, write, mkdirs, posixpath.join, None


@pytest.fixture(scope="module")
def gcs_fs():
    if should_skip(GCS_URI):
        yield [None] * NUM_ATR_FS
        return

    import tensorflow_io_plugin_gs
    from google.cloud import storage

    monkeypatch = pytest.MonkeyPatch()
    bucket_name = os.environ.get("GCS_TEST_BUCKET")
    bucket = None

    # This means we are running against emulator.
    if bucket_name is None:
        monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://localhost:9099")
        monkeypatch.setenv("CLOUD_STORAGE_EMULATOR_ENDPOINT", "http://localhost:9099")

        bucket_name = f"tf-io-bucket-gs-{int(time.time())}"
        client = storage.Client.create_anonymous_client()
        client.project = "test_project"
        bucket = client.create_bucket(bucket_name)
    else:
        # TODO(vnvo2409): Implement for testing against production scenario
        pass

    def parse(path):
        res = urlparse(path, scheme=GCS_URI, allow_fragments=False)
        return res.path[1:]

    def path_to(*args):
        return f"{GCS_URI}://{bucket_name}/{posixpath.join(ROOT_PREFIX, *args)}"

    def read(path):
        key_name = parse(path)
        blob = bucket.get_blob(key_name)
        return blob.download_as_bytes()

    def write(path, body):
        key_name = parse(path)
        blob = bucket.blob(key_name)
        blob.upload_from_string(body)

    def mkdirs(path):
        if path[-1] != "/":
            path += "/"
        write(path, b"")

    yield path_to, read, write, mkdirs, posixpath.join, None
    monkeypatch.undo()


@pytest.fixture
def fs(request, s3_fs, az_fs, az_dsn_fs, https_fs, gcs_fs):
    path_to, read, write, mkdirs, join, internal = [None] * NUM_ATR_FS
    test_fs_uri = request.param
    real_uri = test_fs_uri
    should_skip(test_fs_uri, check_only=False)

    if test_fs_uri == S3_URI:
        path_to, read, write, mkdirs, join, internal = s3_fs
    elif test_fs_uri == AZ_URI:
        path_to, read, write, mkdirs, join, internal = az_fs
    elif test_fs_uri == AZ_DSN_URI:
        real_uri = AZ_URI
        path_to, read, write, mkdirs, join, internal = az_dsn_fs
    elif test_fs_uri == HTTPS_URI:
        path_to, read, write, mkdirs, join, internal = https_fs
    elif test_fs_uri == GCS_URI:
        path_to, read, write, mkdirs, join, internal = gcs_fs

    path_to_rand = None
    test_patchs = request.getfixturevalue("patchs")
    if (test_fs_uri, test_patchs) in fs.path_to_rand_cache:
        path_to_rand = fs.path_to_rand_cache[(test_fs_uri, test_patchs)]
    else:
        path_to_rand = functools.partial(path_to, str(random.getrandbits(32)))
        mkdirs(path_to_rand(""))
        fs.path_to_rand_cache[(test_fs_uri, test_patchs)] = path_to_rand
    yield real_uri, path_to_rand, read, write, mkdirs, join, internal


fs.path_to_rand_cache = {}


@pytest.mark.parametrize(
    "fs, patchs",
    [(S3_URI, None), (AZ_URI, None), (AZ_DSN_URI, None), (GCS_URI, None)],
    indirect=["fs"],
)
def test_init(fs, patchs, monkeypatch):
    _, path_to, _, _, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)
    assert tf.io.gfile.exists(path_to("")) is True


@pytest.mark.parametrize(
    "fs, patchs",
    [(S3_URI, None), (AZ_URI, None), (AZ_DSN_URI, None), (GCS_URI, None)],
    indirect=["fs"],
)
def test_io_read_file(fs, patchs, monkeypatch):
    _, path_to, _, write, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_io_read_file")
    body = b"abcdefghijklmn"
    write(fname, body)

    assert tf.io.read_file(fname) == body


@pytest.mark.parametrize(
    "fs, patchs",
    [(S3_URI, None), (AZ_URI, None), (AZ_DSN_URI, None), (GCS_URI, None)],
    indirect=["fs"],
)
def test_io_write_file(fs, patchs, monkeypatch):
    _, path_to, read, _, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_io_write_file")
    assert tf.io.gfile.exists(fname) is False

    body = b"abcdefghijklmn"
    tf.io.write_file(fname, body)

    assert read(fname) == body


def get_readable_body(uri):
    if uri != HTTPS_URI:
        num_lines = 10
        base_body = b"abcdefghijklmn\n"
        lines = [base_body] * num_lines
        body = b"".join(lines)
        return body
    else:
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_http", "LICENSE-2.0.txt"
        )
        with open(local_path, "rb") as f:
            return f.read()


@pytest.mark.parametrize(
    "fs, patchs",
    [
        (
            S3_URI,
            # `use_multi_part_download` does not work with `seekable`.
            lambda monkeypatch: monkeypatch.setattr(
                tf.io.gfile.GFile, "seekable", lambda _: False
            ),
        ),
        (AZ_URI, None),
        (HTTPS_URI, None),
        (GCS_URI, None),
    ],
    indirect=["fs"],
)
def test_gfile_GFile_readable(fs, patchs, monkeypatch):
    uri, path_to, _, write, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_gfile_GFile_readable")

    body = get_readable_body(uri)
    lines = body.splitlines(True)
    write(fname, body)

    # Simple
    with tf.io.gfile.GFile(fname, "rb") as f:
        file_read = f.read()
        assert file_read == body

    # Notfound
    # TODO(vnvo2409): `az` should raise `tf.errors.NotFoundError`.
    if uri != AZ_URI:
        with pytest.raises(tf.errors.NotFoundError):
            fname_not_found = fname + "_not_found"
            with tf.io.gfile.GFile(fname_not_found, "rb") as f:
                _ = f.read()

    # Read length
    with tf.io.gfile.GFile(fname, "rb") as f:
        read_length = 10
        file_read = f.read(read_length)
        assert file_read == body[:read_length]

        file_read = f.read()
        assert file_read == body[read_length:]

    # Readline
    with tf.io.gfile.GFile(fname, "rb") as f:
        line_count = 0
        while True:
            line = f.readline()
            if not line:
                break
            line_count += 1
            assert line == lines[line_count - 1]
        assert line_count == len(lines)

    # Readlines
    with tf.io.gfile.GFile(fname, "rb") as f:
        gfile_lines = f.readlines()
        assert gfile_lines == lines

    # Seek/Tell
    with tf.io.gfile.GFile(fname, "rb") as f:
        assert f.size() == len(body)
        if f.seekable():
            seek_size = 15
            read_length = 10
            f.seek(seek_size)
            file_read = f.read(read_length)
            assert f.tell() == seek_size + read_length
            assert file_read == body[seek_size : seek_size + read_length]


@pytest.mark.parametrize(
    "fs, patchs",
    [(S3_URI, None), (AZ_URI, None), (HTTPS_URI, None), (GCS_URI, None)],
    indirect=["fs"],
)
def test_dataset_from_remote_filename(fs, patchs, monkeypatch):
    uri, path_to, _, write, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_dataset_from_remote_filename")

    body = get_readable_body(uri)
    lines = body.splitlines(True)
    write(fname, body)

    # TextLineDataset
    line_dataset = tf.data.TextLineDataset(fname)
    count_line_dataset = 0
    for line in line_dataset:
        assert line == lines[count_line_dataset].rstrip()
        count_line_dataset += 1
    assert count_line_dataset == len(lines)


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_GFile_writable(fs, patchs, monkeypatch):
    uri, path_to, read, _, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_gfile_GFile_writable")
    assert tf.io.gfile.exists(fname) is False

    num_lines = 10
    base_body = b"abcdefghijklmn\n"
    body = base_body * num_lines

    # Simple
    with tf.io.gfile.GFile(fname, "wb") as f:
        f.write(body)
        f.flush()
        assert read(fname) == body

    # Append
    # TODO(vnvo2409): implement `az` appendable file.
    if uri != AZ_URI:
        with tf.io.gfile.GFile(fname, "ab") as f:
            f.write(base_body)
            f.flush()
            assert read(fname) == body + base_body


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_isdir(fs, patchs, monkeypatch):
    _, path_to, _, write, mkdirs, join, _ = fs
    mock_patchs(monkeypatch, patchs)

    root_path = "test_gfile_isdir"
    dname = path_to(root_path)
    fname = join(dname, "fname")

    mkdirs(dname)
    write(fname, b"123456789")

    assert tf.io.gfile.isdir(dname) is True
    assert tf.io.gfile.isdir(fname) is False


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_listdir(fs, patchs, monkeypatch):
    uri, path_to, _, write, mkdirs, join, _ = fs
    mock_patchs(monkeypatch, patchs)

    root_path = "test_gfile_listdir"
    dname = path_to(root_path)
    mkdirs(dname)

    num_childs = 5
    childrens = [None] * num_childs
    childrens[0] = join(dname, "subdir")
    # TODO(vnvo2409): `gs` filesystem requires `/` at the end of directory's path.
    # Consider if we could change the behavior for matching the other filesystems.
    if uri == GCS_URI:
        childrens[0] += "/"
    mkdirs(childrens[0])

    body = b"123456789"
    for i in range(1, num_childs):
        childrens[i] = join(dname, f"child_{i}")
        write(childrens[i], body)
        write(join(childrens[0], f"subchild_{i}"), body)

    entries = tf.io.gfile.listdir(dname)
    assert sorted(childrens) == sorted([join(dname, entry) for entry in entries])


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_makedirs(fs, patchs, monkeypatch):
    _, path_to, _, write, _, join, _ = fs
    mock_patchs(monkeypatch, patchs)

    root_path = "test_gfile_makedirs/"
    dname = path_to(root_path)
    subdname = join(dname, "subdir_1")
    assert tf.io.gfile.exists(dname) is False
    assert tf.io.gfile.exists(subdname) is False

    tf.io.gfile.mkdir(subdname)
    write(join(subdname, "fname"), b"123456789")
    assert tf.io.gfile.isdir(subdname) is True


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_remove(fs, patchs, monkeypatch):
    _, path_to, read, write, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_gfile_remove")

    body = b"123456789"
    write(fname, body)

    tf.io.gfile.remove(fname)
    assert tf.io.gfile.exists(fname) is False

    with pytest.raises(Exception):
        read(fname)


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_rmtree(fs, patchs, monkeypatch):
    _, path_to, _, write, mkdirs, join, _ = fs
    mock_patchs(monkeypatch, patchs)

    num_entries = 3
    trees = [path_to("test_gfile_rmtree")] * num_entries
    mkdirs(trees[0])

    for i in range(1, num_entries - 1):
        trees[i] = join(trees[i - 1], f"subdir_{i}")
        mkdirs(trees[i])

    trees[-1] = join(trees[-2], "fname")
    write(trees[-1], b"123456789")

    tf.io.gfile.rmtree(trees[0])

    assert [tf.io.gfile.exists(entry) for entry in trees] == [False] * num_entries


# TODO(vnvo2409): `az` copy operations causes an infinite loop.
@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_copy(fs, patchs, monkeypatch):
    _, path_to, read, write, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    src = path_to("test_gfile_copy_src")
    dst = path_to("test_gfile_copy_dst")

    body = b"123456789"
    write(src, body)

    tf.io.gfile.copy(src, dst)
    assert read(dst) == body

    new_body = body + body.capitalize()
    write(src, new_body)
    with pytest.raises(tf.errors.AlreadyExistsError):
        tf.io.gfile.copy(src, dst)
    assert read(dst) == body

    tf.io.gfile.copy(src, dst, overwrite=True)
    assert read(dst) == new_body


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_rename(fs, patchs, monkeypatch):
    _, path_to, read, write, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    src = path_to("test_gfile_rename_src")
    dst = path_to("test_gfile_rename_dst")

    body = b"123456789"
    write(src, body)

    tf.io.gfile.rename(src, dst)
    assert read(dst) == body
    assert tf.io.gfile.exists(src) is False

    new_body = body + body.capitalize()
    write(src, new_body)
    with pytest.raises(tf.errors.AlreadyExistsError):
        tf.io.gfile.rename(src, dst)
    assert read(dst) == body

    tf.io.gfile.rename(src, dst, overwrite=True)
    assert read(dst) == new_body


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None), (GCS_URI, None)], indirect=["fs"]
)
def test_gfile_glob(fs, patchs, monkeypatch):
    _, path_to, _, write, _, join, _ = fs
    mock_patchs(monkeypatch, patchs)

    dname = path_to("test_gfile_glob/")

    num_items = 3
    childs = [None] * 3
    for ext in ["txt", "md"]:
        for i in range(num_items):
            fname = join(dname, f"{i}.{ext}")
            if ext == "txt":
                childs[i] = fname
            write(fname, b"123456789")

    txt_files = tf.io.gfile.glob(join(dname, "*.txt"))
    assert sorted(txt_files) == sorted(childs)
