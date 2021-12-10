# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A test bench for the Google Cloud Storage C++ Client Library."""

import argparse
import error_response
import flask
import gcs_bucket
import gcs_object
import json
import os
import re
import testbench_utils
import time
import sys
from werkzeug import serving
from werkzeug.middleware.dispatcher import DispatcherMiddleware


root = flask.Flask(__name__, subdomain_matching=True)
root.debug = True


@root.route("/")
def index():
    """Default handler for the test bench."""
    return "OK"


@root.route("/<path:object_name>", subdomain="<bucket_name>")
def root_get_object(bucket_name, object_name):
    return xml_get_object(bucket_name, object_name)


@root.route("/<bucket_name>/<path:object_name>", subdomain="")
def root_get_object_with_bucket(bucket_name, object_name):
    return xml_get_object(bucket_name, object_name)


@root.route("/<path:object_name>", subdomain="<bucket_name>", methods=["PUT"])
def root_put_object(bucket_name, object_name):
    return xml_put_object(flask.request.host_url, bucket_name, object_name)


@root.route("/<bucket_name>/<path:object_name>", subdomain="", methods=["PUT"])
def root_put_object_with_bucket(bucket_name, object_name):
    return xml_put_object(flask.request.host_url, bucket_name, object_name)


@root.errorhandler(error_response.ErrorResponse)
def root_error(error):
    return error.as_response()


# Define the WSGI application to handle bucket requests.
GCS_HANDLER_PATH = "/storage/v1"
gcs = flask.Flask(__name__)
gcs.debug = True


def insert_magic_bucket(base_url):
    if len(testbench_utils.all_buckets()) == 0:
        bucket_name = os.environ.get(
            "GOOGLE_CLOUD_CPP_STORAGE_TEST_BUCKET_NAME", "test-bucket"
        )
        bucket = gcs_bucket.GcsBucket(base_url, bucket_name)
        testbench_utils.insert_bucket(bucket_name, bucket)


@gcs.route("/")
def gcs_index():
    """The default handler for GCS requests."""
    return "OK"


@gcs.errorhandler(error_response.ErrorResponse)
def gcs_error(error):
    return error.as_response()


@gcs.route("/b")
def buckets_list():
    """Implement the 'Buckets: list' API: return the Buckets in a project."""
    base_url = flask.url_for("gcs_index", _external=True)
    project = flask.request.args.get("project")
    if project is None or project.endswith("-"):
        raise error_response.ErrorResponse(
            "Invalid or missing project id in `Buckets: list`"
        )
    insert_magic_bucket(base_url)
    result = {"next_page_token": "", "items": []}
    for name, b in testbench_utils.all_buckets():
        result["items"].append(b.metadata)
    return testbench_utils.filtered_response(flask.request, result)


@gcs.route("/b", methods=["POST"])
def buckets_insert():
    """Implement the 'Buckets: insert' API: create a new Bucket."""
    base_url = flask.url_for("gcs_index", _external=True)
    insert_magic_bucket(base_url)
    payload = json.loads(flask.request.data)
    bucket_name = payload.get("name")
    if bucket_name is None:
        raise error_response.ErrorResponse(
            "Missing bucket name in `Buckets: insert`", status_code=412
        )
    if testbench_utils.has_bucket(bucket_name):
        raise error_response.ErrorResponse(
            "Bucket %s already exists" % bucket_name, status_code=400
        )
    bucket = gcs_bucket.GcsBucket(base_url, bucket_name)
    testbench_utils.insert_bucket(bucket_name, bucket)
    return testbench_utils.filtered_response(flask.request, bucket.metadata)


@gcs.route("/b/<bucket_name>")
def buckets_get(bucket_name):
    """Implement the 'Buckets: get' API: return the metadata for a bucket."""
    base_url = flask.url_for("gcs_index", _external=True)
    insert_magic_bucket(base_url)
    bucket = testbench_utils.lookup_bucket(bucket_name)
    bucket.check_preconditions(flask.request)
    return testbench_utils.filtered_response(flask.request, bucket.metadata)


@gcs.route("/b/<bucket_name>", methods=["DELETE"])
def buckets_delete(bucket_name):
    """Implement the 'Buckets: delete' API."""
    bucket = testbench_utils.lookup_bucket(bucket_name)
    bucket.check_preconditions(flask.request)
    testbench_utils.delete_bucket(bucket_name)
    return testbench_utils.filtered_response(flask.request, {})


@gcs.route("/b/<bucket_name>/o")
def objects_list(bucket_name):
    """Implement the 'Objects: list' API: return the objects in a bucket."""
    # Lookup the bucket, if this fails the bucket does not exist, and this
    # function should return an error.
    base_url = flask.url_for("gcs_index", _external=True)
    insert_magic_bucket(base_url)
    _ = testbench_utils.lookup_bucket(bucket_name)
    result = {"next_page_token": "", "items": [], "prefixes:": []}
    versions_parameter = flask.request.args.get("versions")
    all_versions = versions_parameter is not None and bool(versions_parameter)
    prefixes = set()
    prefix = flask.request.args.get("prefix", "", str)
    delimiter = flask.request.args.get("delimiter", "", str)
    start_offset = flask.request.args.get("startOffset", "", str)
    end_offset = flask.request.args.get("endOffset", "", str)
    bucket_link = bucket_name + "/o/"
    for name, o in testbench_utils.all_objects():
        if name.find(bucket_link + prefix) != 0:
            continue
        if o.get_latest() is None:
            continue
        # We assume `delimiter` has only one character.
        if name[len(bucket_link) :] < start_offset:
            continue
        if end_offset != "" and name[len(bucket_link) :] >= end_offset:
            continue
        delimiter_index = name.find(delimiter, len(bucket_link + prefix))
        if delimiter != "" and delimiter_index > 0:
            # We don't want to include `bucket_link` in the returned prefix.
            prefixes.add(name[len(bucket_link) : delimiter_index + 1])
            continue
        if all_versions:
            for object_version in o.revisions.values():
                result["items"].append(object_version.metadata)
        else:
            result["items"].append(o.get_latest().metadata)
    result["prefixes"] = list(prefixes)
    return testbench_utils.filtered_response(flask.request, result)


@gcs.route(
    "/b/<source_bucket>/o/<path:source_object>/copyTo/b/<destination_bucket>/o/<path:destination_object>",
    methods=["POST"],
)
def objects_copy(source_bucket, source_object, destination_bucket, destination_object):
    """Implement the 'Objects: copy' API, copy an object."""
    object_path, blob = testbench_utils.lookup_object(source_bucket, source_object)
    blob.check_preconditions(
        flask.request,
        if_generation_match="ifSourceGenerationMatch",
        if_generation_not_match="ifSourceGenerationNotMatch",
        if_metageneration_match="ifSourceMetagenerationMatch",
        if_metageneration_not_match="ifSourceMetagenerationNotMatch",
    )
    source_revision = blob.get_revision(flask.request, "sourceGeneration")
    if source_revision is None:
        raise error_response.ErrorResponse(
            "Revision not found %s" % object_path, status_code=404
        )

    destination_path, destination = testbench_utils.get_object(
        destination_bucket,
        destination_object,
        gcs_object.GcsObject(destination_bucket, destination_object),
    )
    base_url = flask.url_for("gcs_index", _external=True)
    current_version = destination.copy_from(base_url, flask.request, source_revision)
    testbench_utils.insert_object(destination_path, destination)
    return testbench_utils.filtered_response(flask.request, current_version.metadata)


@gcs.route(
    "/b/<source_bucket>/o/<path:source_object>/rewriteTo/b/<destination_bucket>/o/<path:destination_object>",
    methods=["POST"],
)
def objects_rewrite(
    source_bucket, source_object, destination_bucket, destination_object
):
    """Implement the 'Objects: rewrite' API."""
    base_url = flask.url_for("gcs_index", _external=True)
    insert_magic_bucket(base_url)
    object_path, blob = testbench_utils.lookup_object(source_bucket, source_object)
    blob.check_preconditions(
        flask.request,
        if_generation_match="ifSourceGenerationMatch",
        if_generation_not_match="ifSourceGenerationNotMatch",
        if_metageneration_match="ifSourceMetagenerationMatch",
        if_metageneration_not_match="ifSourceMetagenerationNotMatch",
    )
    response = blob.rewrite_step(
        base_url, flask.request, destination_bucket, destination_object
    )
    return testbench_utils.filtered_response(flask.request, response)


def objects_get_common(bucket_name, object_name, revision):
    # Respect the Range: header, if present.
    range_header = flask.request.headers.get("range")
    response_payload = revision.media
    begin = 0
    end = len(response_payload)
    if range_header is not None:
        m = re.match("bytes=([0-9]+)-([0-9]+)", range_header)
        if m:
            begin = int(m.group(1))
            end = int(m.group(2))
            response_payload = response_payload[begin : end + 1]
        m = re.match("bytes=([0-9]+)-$", range_header)
        if m:
            begin = int(m.group(1))
            response_payload = response_payload[begin:]
        m = re.match("bytes=-([0-9]+)$", range_header)
        if m:
            last = int(m.group(1))
            response_payload = response_payload[-last:]
    # Process custom headers to test error conditions.
    instructions = flask.request.headers.get("x-goog-testbench-instructions")
    if instructions == "return-broken-stream":

        def streamer():
            chunk_size = 64 * 1024
            for r in range(0, len(response_payload), chunk_size):
                if r > 1024 * 1024:
                    print("\n\n###### EXIT to simulate crash\n")
                    sys.exit(1)
                time.sleep(0.1)
                chunk_end = min(r + chunk_size, len(response_payload))
                yield response_payload[r:chunk_end]

        length = len(response_payload)
        content_range = "bytes %d-%d/%d" % (begin, end - 1, length)
        headers = {
            "Content-Range": content_range,
            "Content-Length": length,
            "x-goog-hash": revision.x_goog_hash_header(),
            "x-goog-generation": revision.generation,
        }
        return flask.Response(streamer(), status=200, headers=headers)

    if instructions == "return-corrupted-data":
        response_payload = testbench_utils.corrupt_media(response_payload)

    if instructions is not None and instructions.startswith("stall-always"):
        length = len(response_payload)
        content_range = "bytes %d-%d/%d" % (begin, end - 1, length)

        def streamer():
            chunk_size = 16 * 1024
            for r in range(begin, end, chunk_size):
                chunk_end = min(r + chunk_size, end)
                if r == begin:
                    time.sleep(10)
                yield response_payload[r:chunk_end]

        headers = {
            "Content-Range": content_range,
            "x-goog-hash": revision.x_goog_hash_header(),
            "x-goog-generation": revision.generation,
        }
        return flask.Response(streamer(), status=200, headers=headers)

    if instructions == "stall-at-256KiB" and begin == 0:
        length = len(response_payload)
        content_range = "bytes %d-%d/%d" % (begin, end - 1, length)

        def streamer():
            chunk_size = 16 * 1024
            for r in range(begin, end, chunk_size):
                chunk_end = min(r + chunk_size, end)
                if r == 256 * 1024:
                    time.sleep(10)
                yield response_payload[r:chunk_end]

        headers = {
            "Content-Range": content_range,
            "x-goog-hash": revision.x_goog_hash_header(),
            "x-goog-generation": revision.generation,
        }
        return flask.Response(streamer(), status=200, headers=headers)

    if instructions is not None and instructions.startswith("return-503-after-256K"):
        length = len(response_payload)
        headers = {
            "Content-Range": "bytes %d-%d/%d" % (begin, end - 1, length),
            "x-goog-hash": revision.x_goog_hash_header(),
            "x-goog-generation": revision.generation,
        }
        if begin == 0:

            def streamer():
                chunk_size = 4 * 1024
                for r in range(0, len(response_payload), chunk_size):
                    if r >= 256 * 1024:
                        print("\n\n###### EXIT to simulate crash\n")
                        sys.exit(1)
                    time.sleep(0.01)
                    chunk_end = min(r + chunk_size, len(response_payload))
                    yield response_payload[r:chunk_end]

            return flask.Response(streamer(), status=200, headers=headers)
        if instructions.endswith("/retry-1"):
            print("## Return error for retry 1")
            return flask.Response("Service Unavailable", status=503)
        if instructions.endswith("/retry-2"):
            print("## Return error for retry 2")
            return flask.Response("Service Unavailable", status=503)
        print("## Return success for %s" % instructions)
        return flask.Response(response_payload, status=200, headers=headers)

    response = flask.make_response(response_payload)
    length = len(response_payload)
    content_range = "bytes %d-%d/%d" % (begin, end - 1, length)
    response.headers["Content-Range"] = content_range
    response.headers["x-goog-hash"] = revision.x_goog_hash_header()
    response.headers["x-goog-generation"] = revision.generation
    return response


@gcs.route("/b/<bucket_name>/o/<path:object_name>", methods=["DELETE"])
def objects_delete(bucket_name, object_name):
    """Implement the 'Objects: delete' API.  Delete objects."""
    object_path, blob = testbench_utils.lookup_object(bucket_name, object_name)
    blob.check_preconditions(flask.request)
    remove = blob.del_revision(flask.request)
    if remove:
        testbench_utils.delete_object(object_path)
    return testbench_utils.filtered_response(flask.request, {})


@gcs.route("/b/<bucket_name>/o/<path:object_name>/compose", methods=["POST"])
def objects_compose(bucket_name, object_name):
    """Implement the 'Objects: compose' API: concatenate Objects."""
    payload = json.loads(flask.request.data)
    source_objects = payload["sourceObjects"]
    if source_objects is None:
        raise error_response.ErrorResponse(
            "You must provide at least one source component.", status_code=400
        )
    if len(source_objects) > 32:
        raise error_response.ErrorResponse(
            "The number of source components provided"
            " (%d) exceeds the maximum (32)" % len(source_objects),
            status_code=400,
        )
    composed_media = b""
    for source_object in source_objects:
        source_object_name = source_object.get("name")
        if source_object_name is None:
            raise error_response.ErrorResponse("Required.", status_code=400)
        source_object_path, source_blob = testbench_utils.lookup_object(
            bucket_name, source_object_name
        )
        source_revision = source_blob.get_latest()
        generation = source_object.get("generation")
        if generation is not None:
            source_revision = source_blob.get_revision_by_generation(generation)
            if source_revision is None:
                raise error_response.ErrorResponse(
                    "No such object: %s" % source_object_path, status_code=404
                )
        object_preconditions = source_object.get("objectPreconditions")
        if object_preconditions is not None:
            if_generation_match = object_preconditions.get("ifGenerationMatch")
            source_blob.check_preconditions_by_value(
                if_generation_match, None, None, None
            )
        composed_media += source_revision.media
    composed_object_path, composed_object = testbench_utils.get_object(
        bucket_name, object_name, gcs_object.GcsObject(bucket_name, object_name)
    )
    composed_object.check_preconditions(flask.request)
    base_url = flask.url_for("gcs_index", _external=True)
    current_version = composed_object.compose_from(
        base_url, flask.request, composed_media
    )
    testbench_utils.insert_object(composed_object_path, composed_object)
    return testbench_utils.filtered_response(flask.request, current_version.metadata)


# Define the WSGI application to handle bucket requests.
DOWNLOAD_HANDLER_PATH = "/download/storage/v1"
download = flask.Flask(__name__)
download.debug = True


@download.errorhandler(error_response.ErrorResponse)
def download_error(error):
    return error.as_response()


@gcs.route("/b/<bucket_name>/o/<path:object_name>")
@download.route("/b/<bucket_name>/o/<path:object_name>")
def objects_get(bucket_name, object_name):
    """Implement the 'Objects: get' API.  Read objects or their metadata."""
    _, blob = testbench_utils.lookup_object(bucket_name, object_name)
    blob.check_preconditions(flask.request)
    revision = blob.get_revision(flask.request)

    media = flask.request.args.get("alt", None)
    if media is None or media == "json":
        return testbench_utils.filtered_response(flask.request, revision.metadata)
    if media != "media":
        raise error_response.ErrorResponse("Invalid alt=%s parameter" % media)
    revision.validate_encryption_for_read(flask.request)
    return objects_get_common(bucket_name, object_name, revision)


# Define the WSGI application to handle bucket requests.
UPLOAD_HANDLER_PATH = "/upload/storage/v1"
upload = flask.Flask(__name__)
upload.debug = True


@upload.errorhandler(error_response.ErrorResponse)
def upload_error(error):
    return error.as_response()


@upload.route("/b/<bucket_name>/o", methods=["POST"])
def objects_insert(bucket_name):
    """Implement the 'Objects: insert' API.  Insert a new GCS Object."""
    gcs_url = flask.url_for(
        "objects_insert", bucket_name=bucket_name, _external=True
    ).replace("/upload/", "/")
    insert_magic_bucket(gcs_url)

    upload_type = flask.request.args.get("uploadType")
    if upload_type is None:
        raise error_response.ErrorResponse(
            "uploadType not set in Objects: insert", status_code=400
        )
    if upload_type not in {"multipart", "media", "resumable"}:
        raise error_response.ErrorResponse(
            "testbench does not support %s uploadType" % upload_type, status_code=400
        )

    if upload_type == "resumable":
        bucket = testbench_utils.lookup_bucket(bucket_name)
        upload_url = flask.url_for(
            "objects_insert", bucket_name=bucket_name, _external=True
        )
        return bucket.create_resumable_upload(upload_url, flask.request)

    object_path = None
    blob = None
    current_version = None
    if upload_type == "media":
        object_name = flask.request.args.get("name", None)
        if object_name is None:
            raise error_response.ErrorResponse(
                "name not set in Objects: insert", status_code=412
            )
        object_path, blob = testbench_utils.get_object(
            bucket_name, object_name, gcs_object.GcsObject(bucket_name, object_name)
        )
        blob.check_preconditions(flask.request)
        current_version = blob.insert(gcs_url, flask.request)
    else:
        resource, media_headers, media_body = testbench_utils.parse_multi_part(
            flask.request
        )
        object_name = flask.request.args.get("name", resource.get("name", None))
        if object_name is None:
            raise error_response.ErrorResponse(
                "name not set in Objects: insert", status_code=412
            )
        object_path, blob = testbench_utils.get_object(
            bucket_name, object_name, gcs_object.GcsObject(bucket_name, object_name)
        )
        blob.check_preconditions(flask.request)
        current_version = blob.insert_multipart(
            gcs_url, flask.request, resource, media_headers, media_body
        )
    testbench_utils.insert_object(object_path, blob)
    return testbench_utils.filtered_response(flask.request, current_version.metadata)


@upload.route("/b/<bucket_name>/o", methods=["PUT"])
def resumable_upload_chunk(bucket_name):
    """Receive a chunk for a resumable upload."""
    gcs_url = flask.url_for(
        "objects_insert", bucket_name=bucket_name, _external=True
    ).replace("/upload/", "/")
    bucket = testbench_utils.lookup_bucket(bucket_name)
    return bucket.receive_upload_chunk(gcs_url, flask.request)


@upload.route("/b/<bucket_name>/o", methods=["DELETE"])
def delete_resumable_upload(bucket_name):
    upload_type = flask.request.args.get("uploadType")
    if upload_type != "resumable":
        raise error_response.ErrorResponse(
            "testbench can delete resumable uploadType only", status_code=400
        )
    upload_id = flask.request.args.get("upload_id")
    if upload_id is None:
        raise error_response.ErrorResponse(
            "missing upload_id in delete_resumable_upload", status_code=400
        )
    bucket = testbench_utils.lookup_bucket(bucket_name)
    if upload_id not in bucket.resumable_uploads:
        raise error_response.ErrorResponse("upload_id does not exist", status_code=404)
    bucket.resumable_uploads.pop(upload_id)
    return testbench_utils.filtered_response(flask.request, {})


def xml_put_object(gcs_url, bucket_name, object_name):
    """Implement PUT for the XML API."""
    insert_magic_bucket(gcs_url)
    object_path, blob = testbench_utils.get_object(
        bucket_name, object_name, gcs_object.GcsObject(bucket_name, object_name)
    )
    generation_match = flask.request.headers.get("x-goog-if-generation-match")
    metageneration_match = flask.request.headers.get("x-goog-if-metageneration-match")
    blob.check_preconditions_by_value(
        generation_match, None, metageneration_match, None
    )
    revision = blob.insert_xml(gcs_url, flask.request)
    testbench_utils.insert_object(object_path, blob)
    response = flask.make_response("")
    response.headers["x-goog-hash"] = revision.x_goog_hash_header()
    return response


def xml_get_object(bucket_name, object_name):
    """Implement the 'Objects: insert' API.  Insert a new GCS Object."""
    object_path, blob = testbench_utils.lookup_object(bucket_name, object_name)
    if flask.request.args.get("acl") is not None:
        raise error_response.ErrorResponse(
            "ACL query not supported in XML API", status_code=500
        )
    if flask.request.args.get("encryption") is not None:
        raise error_response.ErrorResponse(
            "Encryption query not supported in XML API", status_code=500
        )
    generation_match = flask.request.headers.get("if-generation-match")
    metageneration_match = flask.request.headers.get("if-metageneration-match")
    blob.check_preconditions_by_value(
        generation_match, None, metageneration_match, None
    )
    revision = blob.get_revision(flask.request)
    return objects_get_common(bucket_name, object_name, revision)


application = DispatcherMiddleware(
    root,
    {
        GCS_HANDLER_PATH: gcs,
        UPLOAD_HANDLER_PATH: upload,
        DOWNLOAD_HANDLER_PATH: download,
    },
)
