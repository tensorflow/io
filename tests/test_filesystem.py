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

import os
import posixpath
import sys
import time
from urllib.parse import urlparse

import boto3
import pytest
import tensorflow as tf
import tensorflow_io as tfio  # pylint: disable=unused-import
from azure.storage.blob import ContainerClient

pytestmark = pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason="TODO emulator not setup properly on macOS/Windows yet",
)

ROOT_PREFIX = f"tf-io-root-{int(time.time())}/"
S3_URI = "s3"
AZ_URI = "az"


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


@pytest.fixture(scope="module")
def s3_fs():
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

    yield S3_URI, path_to, read, write, mkdirs, posixpath.join
    monkeypatch.undo()


@pytest.fixture(scope="module")
def az_fs():
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

    def mkdirs(_):
        pass

    yield AZ_URI, path_to, read, write, mkdirs, posixpath.join
    monkeypatch.undo()


@pytest.fixture
def fs(request, s3_fs, az_fs):
    if request.param == S3_URI:
        return s3_fs
    elif request.param == AZ_URI:
        return az_fs


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_init(fs, patchs, monkeypatch):
    _, path_to, _, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)
    assert tf.io.gfile.exists(path_to("")) is True


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_io_read_file(fs, patchs, monkeypatch):
    _, path_to, _, write, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_io_read_file")
    body = b"abcdefghijklmn"
    write(fname, body)

    assert tf.io.read_file(fname) == body


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_io_write_file(fs, patchs, monkeypatch):
    _, path_to, read, _, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_io_write_file")
    assert tf.io.gfile.exists(fname) is False

    body = b"abcdefghijklmn"
    tf.io.write_file(fname, body)

    assert read(fname) == body


@pytest.mark.parametrize(
    "fs, patchs",
    [
        (
            S3_URI,
            # `use_multi_part_download` does not work with `seakable`.
            lambda monkeypatch: monkeypatch.setattr(
                tf.io.gfile.GFile, "seekable", lambda _: False
            ),
        ),
        (AZ_URI, None),
    ],
    indirect=["fs"],
)
def test_gfile_GFile_readable(fs, patchs, monkeypatch):
    uri, path_to, _, write, _, _ = fs
    mock_patchs(monkeypatch, patchs)

    fname = path_to("test_gfile_GFile_readable")

    num_lines = 10
    base_body = b"abcdefghijklmn\n"
    body = base_body * num_lines
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
            assert line == base_body
        assert line_count == num_lines

    # Readlines
    with tf.io.gfile.GFile(fname, "rb") as f:
        lines = f.readlines()
        assert lines == [base_body] * num_lines

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
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_gfile_GFile_writable(fs, patchs, monkeypatch):
    uri, path_to, read, _, _, _ = fs
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
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_gfile_isdir(fs, patchs, monkeypatch):
    _, path_to, _, write, mkdirs, join = fs
    mock_patchs(monkeypatch, patchs)

    root_path = "test_gfile_isdir"
    dname = path_to(root_path)
    fname = join(dname, "fname")

    mkdirs(dname)
    write(fname, b"123456789")

    assert tf.io.gfile.isdir(dname) is True
    assert tf.io.gfile.isdir(fname) is False


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_gfile_listdir(fs, patchs, monkeypatch):
    _, path_to, _, write, mkdirs, join = fs
    mock_patchs(monkeypatch, patchs)

    root_path = "test_gfile_listdir"
    dname = path_to(root_path)
    mkdirs(dname)

    num_childs = 5
    childrens = [None] * num_childs
    childrens[0] = join(dname, "subdir")
    mkdirs(childrens[0])

    body = b"123456789"
    for i in range(1, num_childs):
        childrens[i] = join(dname, f"child_{i}")
        write(childrens[i], body)
        write(join(childrens[0], f"subchild_{i}"), body)

    entries = tf.io.gfile.listdir(dname)
    assert sorted(childrens) == sorted([join(dname, entry) for entry in entries])


@pytest.mark.parametrize(
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_gfile_makedirs(fs, patchs, monkeypatch):
    _, path_to, _, write, _, join = fs
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
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_gfile_rmtree(fs, patchs, monkeypatch):
    _, path_to, _, write, mkdirs, join = fs
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
@pytest.mark.parametrize("fs, patchs", [(S3_URI, None)], indirect=["fs"])
def test_gfile_copy(fs, patchs, monkeypatch):
    _, path_to, read, write, _, _ = fs
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
    "fs, patchs", [(S3_URI, None), (AZ_URI, None)], indirect=["fs"]
)
def test_gfile_glob(fs, patchs, monkeypatch):
    _, path_to, _, write, _, join = fs
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
