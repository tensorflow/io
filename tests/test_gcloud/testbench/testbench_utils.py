# Copyright 2018 Google LLC
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
"""Standalone helpers for the Google Cloud Storage test bench."""

import base64
import error_response
import hashlib
import json
import random
import re

field_match = re.compile(r"(?:(\w+)\((\w+(?:,\w+)*)\))|(\w+)")


def filter_fields_from_response(fields, response):
    """Format the response as a JSON string, using any filtering included in
    the request.

    :param fields:str the value of the `fields` parameter in the original
        request.
    :param response:dict a dictionary to be formatted as a JSON string.
    :return: the response formatted as a string.
    :rtype:str
    """
    if fields is None:
        return json.dumps(response)
    tmp = {}
    fields.replace(" ", "")
    for keys in field_match.findall(fields):
        if keys[2]:
            if keys[2] not in response:
                continue
            tmp[keys[2]] = response[keys[2]]
        else:
            if keys[0] not in response:
                continue
            childrens = response[keys[0]]
            if isinstance(childrens, list):
                tmp_list = []
                for children in childrens:
                    child = {}
                    for child_key in keys[1].split(","):
                        child[child_key] = children[child_key]
                    tmp_list.append(child)
                tmp[keys[0]] = tmp_list
            elif isinstance(childrens, dict):
                child = {}
                for child_key in keys[1].split(","):
                    child[child_key] = children[child_key]
                tmp[keys[0]] = child
    return json.dumps(tmp)


def filtered_response(request, response):
    """Format the response as a JSON string, using any filtering included in
    the request.

    :param request:flask.Request the original HTTP request.
    :param response:dict a dictionary to be formatted as a JSON string.
    :return: the response formatted as a string.
    :rtype:str
    """
    fields = request.args.get("fields")
    return filter_fields_from_response(fields, response)


def raise_csek_error(code=400):
    msg = "Missing a SHA256 hash of the encryption key, or it is not"
    msg += " base64 encoded, or it does not match the encryption key."
    link = "https://cloud.google.com/storage/docs/encryption#customer-supplied_encryption_keys"
    error = {
        "error": {
            "errors": [
                {
                    "domain": "global",
                    "reason": "customerEncryptionKeySha256IsInvalid",
                    "message": msg,
                    "extendedHelp": link,
                }
            ],
            "code": code,
            "message": msg,
        }
    }
    raise error_response.ErrorResponse(json.dumps(error), status_code=code)


def validate_customer_encryption_headers(
    key_header_value, hash_header_value, algo_header_value
):
    """Verify that the encryption headers are internally consistent.

    :param key_header_value: str the value of the x-goog-*-key header
    :param hash_header_value: str the value of the x-goog-*-key-sha256 header
    :param algo_header_value: str the value of the x-goog-*-key-algorithm header
    :rtype: NoneType
    """
    try:
        if algo_header_value is None or algo_header_value != "AES256":
            raise error_response.ErrorResponse(
                "Invalid or missing algorithm %s for CSEK" % algo_header_value,
                status_code=400,
            )

        key = base64.standard_b64decode(key_header_value)
        if key is None or len(key) != 256 / 8:
            raise_csek_error()

        h = hashlib.sha256()
        h.update(key)
        expected = base64.standard_b64encode(h.digest()).decode("utf-8")
        if hash_header_value is None or expected != hash_header_value:
            raise_csek_error()
    except error_response.ErrorResponse:
        # error_response.ErrorResponse indicates that the request was invalid, just pass
        # that exception through.
        raise
    except Exception:
        # Many of the functions above may raise, convert those to an
        # error_response.ErrorResponse with the right format.
        raise_csek_error()


def extract_media(request):
    """Extract the media from a flask Request.

    To avoid race conditions when using greenlets we cannot perform I/O in the
    constructor of GcsObjectVersion, or in any of the operations that modify
    the state of the service.  Because sometimes the media is uploaded with
    chunked encoding, we need to do I/O before finishing the GcsObjectVersion
    creation. If we do this I/O after the GcsObjectVersion creation started,
    the the state of the application may change due to other I/O.

    :param request:flask.Request the HTTP request.
    :return: the full media of the request.
    :rtype: str
    """
    if request.environ.get("HTTP_TRANSFER_ENCODING", "") == "chunked":
        return request.environ.get("wsgi.input").read()
    return request.data


def corrupt_media(media):
    """Return a randomly modified version of a string.

    :param media:bytes a string (typically some object media) to be modified.
    :return: a string that is slightly different than media.
    :rtype: str
    """
    # Deal with the boundary condition.
    if not media:
        return bytearray(random.sample("abcdefghijklmnopqrstuvwxyz", 1), "utf-8")
    return b"B" + media[1:] if media[0:1] == b"A" else b"A" + media[1:]


# Define the collection of Buckets indexed by <bucket_name>
GCS_BUCKETS = dict()


def lookup_bucket(bucket_name):
    """Lookup a bucket by name in the global collection.

    :param bucket_name:str the name of the Bucket.
    :return: the bucket matching the name.
    :rtype:GcsBucket
    :raises:ErrorResponse if the bucket is not found.
    """
    bucket = GCS_BUCKETS.get(bucket_name)
    if bucket is None:
        raise error_response.ErrorResponse(
            "Bucket %s not found" % bucket_name, status_code=404
        )
    return bucket


def has_bucket(bucket_name):
    """Return True if the bucket already exists in the global collection."""
    return GCS_BUCKETS.get(bucket_name) is not None


def insert_bucket(bucket_name, bucket):
    """Insert (or replace) a new bucket into the global collection.

    :param bucket_name:str the name of the bucket.
    :param bucket:GcsBucket the bucket to insert.
    """
    GCS_BUCKETS[bucket_name] = bucket


def delete_bucket(bucket_name):
    """Delete a bucket from the global collection."""
    GCS_BUCKETS.pop(bucket_name)


def all_buckets():
    """Return a key,value iterator for all the buckets in the global collection.

    :rtype:dict[str, GcsBucket]
    """
    return GCS_BUCKETS.items()


# Define the collection of GcsObjects indexed by <bucket_name>/o/<object_name>
GCS_OBJECTS = dict()


def lookup_object(bucket_name, object_name):
    """Lookup an object by name in the global collection.

    :param bucket_name:str the name of the Bucket that contains the object.
    :param object_name:str the name of the Object.
    :return: tuple the object path and the object.
    :rtype: (str,GcsObject)
    :raises:ErrorResponse if the object is not found.
    """
    object_path, gcs_object = get_object(bucket_name, object_name, None)
    if gcs_object is None:
        raise error_response.ErrorResponse(
            f"Object {object_name} in {bucket_name} not found",
            status_code=404,
        )
    return object_path, gcs_object


def get_object(bucket_name, object_name, default_value):
    """Find an object in the global collection, return a default value if not
    found.

    :param bucket_name:str the name of the Bucket that contains the object.
    :param object_name:str the name of the Object.
    :param default_value:GcsObject the default value returned if the object is
        not found.
    :return: tuple the object path and the object.
    :rtype: (str,GcsObject)
    """
    object_path = bucket_name + "/o/" + object_name
    return object_path, GCS_OBJECTS.get(object_path, default_value)


def insert_object(object_path, value):
    """Insert an object to the global collection."""
    GCS_OBJECTS[object_path] = value


def delete_object(object_path):
    """Delete an object from the global collection."""
    GCS_OBJECTS.pop(object_path)


def all_objects():
    """Return a key,value iterator for all the objects in the global collection.

    :rtype:dict[str, GcsBucket]
    """
    return GCS_OBJECTS.items()


def parse_multi_part(request):
    """Parse a multi-part request

    :param request:flask.Request multipart request.
    :return: a tuple with the resource, media_headers and the media_body.
    :rtype: (dict, dict, str)
    """
    content_type = request.headers.get("content-type")
    if content_type is None or not content_type.startswith("multipart/related"):
        raise error_response.ErrorResponse(
            "Missing or invalid content-type header in multipart upload"
        )
    _, _, boundary = content_type.partition("boundary=")
    boundary = boundary.strip('"')
    if boundary is None:
        raise error_response.ErrorResponse(
            "Missing or invalid boundary in content-type header in multipart upload"
        )

    def parse_metadata(part):
        result = part.split(b"\r\n")
        if result[0] != b"" and result[-1] != b"":
            raise error_response.ErrorResponse(
                "Missing or invalid multipart %s" % str(part)
            )
        result = list(filter(None, result))
        headers = {}
        if len(result) < 2:
            result.append(b"")
        for header in result[:-1]:
            key, value = header.split(b": ")
            headers[key.decode("utf-8").lower()] = value.decode("utf-8")
        return result[-1]

    def parse_body(part):
        if part[0:2] != b"\r\n" or part[-2:] != b"\r\n":
            raise error_response.ErrorResponse(
                "Missing or invalid multipart %s" % str(part)
            )
        part = part[2:-2]
        part.lstrip(b"\r\n")
        content_type_index = part.find(b"\r\n")
        if content_type_index == -1:
            raise error_response.ErrorResponse(
                "Missing or invalid multipart %s" % str(part)
            )
        content_type = part[:content_type_index]
        _, value = content_type.decode("utf-8").split(": ")
        media = part[content_type_index + 2 :]
        if media[:2] == b"\r\n":
            # It is either `\r\n` or `\r\n\r\n`, we should remove at most 4 characters.
            media = media[2:]
        return {"content-type": value}, media

    boundary = boundary.encode("utf-8")
    body = extract_media(request)
    parts = body.split(b"--" + boundary)
    if parts[-1] != b"--\r\n" and parts[-1] != b"--":
        raise error_response.ErrorResponse(
            "Missing end marker (--%s--) in media body" % boundary
        )
    resource = parse_metadata(parts[1])
    metadata = json.loads(resource)
    content_type, media = parse_body(parts[2])
    return metadata, content_type, media
