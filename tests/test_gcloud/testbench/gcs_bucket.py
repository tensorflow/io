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
"""Implement a class to simulate GCS buckets."""

import base64
import error_response
import flask
import gcs_object
import json
import re
import testbench_utils
import time


class GcsBucket:
    """Represent a GCS Bucket."""

    def __init__(self, gcs_url, name):
        self.name = name
        self.gcs_url = gcs_url
        now = time.gmtime(time.time())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", now)
        self.metadata = {
            "timeCreated": timestamp,
            "updated": timestamp,
            "metageneration": "0",
            "name": self.name,
            "location": "US",
            "storageClass": "STANDARD",
            "etag": "XYZ=",
            "labels": {"foo": "bar", "baz": "qux"},
            "owner": {"entity": "project-owners-123456789", "entityId": ""},
        }
        self.resumable_uploads = {}

    def versioning_enabled(self):
        """Return True if versioning is enabled for this Bucket."""
        v = self.metadata.get("versioning", None)
        if v is None:
            return False
        return v.get("enabled", False)

    def check_preconditions(self, request):
        """Verify that the preconditions in request are met.

        :param request:flask.Request the contents of the HTTP request.
        :rtype:NoneType
        :raises:ErrorResponse if the request does not pass the preconditions,
            for example, the request has a `ifMetagenerationMatch` restriction
            that is not met.
        """

        metageneration_match = request.args.get("ifMetagenerationMatch")
        metageneration_not_match = request.args.get("ifMetagenerationNotMatch")
        metageneration = self.metadata.get("metageneration")

        if (
            metageneration_not_match is not None
            and metageneration_not_match == metageneration
        ):
            raise error_response.ErrorResponse(
                "Precondition Failed (metageneration = %s)" % metageneration,
                status_code=412,
            )

        if metageneration_match is not None and metageneration_match != metageneration:
            raise error_response.ErrorResponse(
                "Precondition Failed (metageneration = %s)" % metageneration,
                status_code=412,
            )

    def create_resumable_upload(self, upload_url, request):
        """Capture the details for a resumable upload.

        :param upload_url: str the base URL for uploads.
        :param request: flask.Request the original http request.
        :return: the HTTP response to send back.
        """
        x_upload_content_type = request.headers.get(
            "x-upload-content-type", "application/octet-stream"
        )
        x_upload_content_length = request.headers.get("x-upload-content-length")
        expected_bytes = None
        if x_upload_content_length:
            expected_bytes = int(x_upload_content_length)

        if request.args.get("name") is not None and len(request.data):
            raise error_response.ErrorResponse(
                "The name argument is only supported for empty payloads",
                status_code=400,
            )
        if len(request.data):
            metadata = json.loads(request.data)
        else:
            metadata = {"name": request.args.get("name")}

        if metadata.get("name") is None:
            raise error_response.ErrorResponse(
                "Missing object name argument", status_code=400
            )
        metadata.setdefault("contentType", x_upload_content_type)
        upload = {
            "metadata": metadata,
            "instructions": request.headers.get("x-goog-testbench-instructions"),
            "fields": request.args.get("fields"),
            "next_byte": 0,
            "expected_bytes": expected_bytes,
            "object_name": metadata.get("name"),
            "media": b"",
            "transfer": set(),
            "done": False,
        }
        # Capture the preconditions, including those that are None.
        for precondition in [
            "ifGenerationMatch",
            "ifGenerationNotMatch",
            "ifMetagenerationMatch",
            "ifMetagenerationNotMatch",
        ]:
            upload[precondition] = request.args.get(precondition)
        upload_id = base64.b64encode(bytearray(metadata.get("name"), "utf-8")).decode(
            "utf-8"
        )
        self.resumable_uploads[upload_id] = upload
        location = f"{upload_url}?uploadType=resumable&upload_id={upload_id}"
        response = flask.make_response("")
        response.headers["Location"] = location
        return response

    def receive_upload_chunk(self, gcs_url, request):
        """Receive a new upload chunk.

        :param gcs_url: str the base URL for the service.
        :param request: flask.Request the original http request.
        :return: the HTTP response.
        """
        upload_id = request.args.get("upload_id")
        if upload_id is None:
            raise error_response.ErrorResponse(
                "Missing upload_id in resumable_upload_chunk", status_code=400
            )
        upload = self.resumable_uploads.get(upload_id)
        if upload is None:
            raise error_response.ErrorResponse(
                "Cannot find resumable upload %s" % upload_id, status_code=404
            )
        # Be gracious in what you accept, if the Content-Range header is not
        # set we assume it is a good header and it is the end of the file.
        next_byte = upload["next_byte"]
        upload["transfer"].add(request.environ.get("HTTP_TRANSFER_ENCODING", ""))
        end = next_byte + len(request.data)
        total = end
        final_chunk = False
        payload = testbench_utils.extract_media(request)
        content_range = request.headers.get("content-range")
        if content_range is not None:
            if content_range.startswith("bytes */*"):
                # This is just a query to resume an upload, if it is done, return
                # the completed upload payload and an empty range header.
                response = flask.make_response(upload.get("payload", ""))
                if next_byte > 1 and not upload["done"]:
                    response.headers["Range"] = "bytes=0-%d" % (next_byte - 1)
                response.status_code = 200 if upload["done"] else 308
                return response
            match = re.match(r"bytes \*/(\*|[0-9]+)", content_range)
            if match:
                if match.group(1) == "*":
                    total = 0
                else:
                    total = int(match.group(1))
                    final_chunk = True
            else:
                match = re.match(r"bytes ([0-9]+)-([0-9]+)\/(\*|[0-9]+)", content_range)
                if not match:
                    raise error_response.ErrorResponse(
                        "Invalid Content-Range in upload %s" % content_range,
                        status_code=400,
                    )
                begin = int(match.group(1))
                end = int(match.group(2))
                if match.group(3) == "*":
                    total = 0
                else:
                    total = int(match.group(3))
                    final_chunk = True

                if begin != next_byte:
                    raise error_response.ErrorResponse(
                        "Mismatched data range, expected data at %d, got %d"
                        % (next_byte, begin),
                        status_code=400,
                    )
                if len(payload) != end - begin + 1:
                    raise error_response.ErrorResponse(
                        "Mismatched data range (%d) vs. received data (%d)"
                        % (end - begin + 1, len(payload)),
                        status_code=400,
                    )

        upload["media"] = upload.get("media", b"") + payload
        next_byte = len(upload.get("media", ""))
        upload["next_byte"] = next_byte
        response_payload = ""
        if final_chunk and next_byte >= total:
            expected_bytes = upload["expected_bytes"]
            if expected_bytes is not None and expected_bytes != total:
                raise error_response.ErrorResponse(
                    "X-Upload-Content-Length"
                    "validation failed. Expected=%d, got %d." % (expected_bytes, total)
                )
            upload["done"] = True
            object_name = upload.get("object_name")
            object_path, blob = testbench_utils.get_object(
                self.name, object_name, gcs_object.GcsObject(self.name, object_name)
            )
            # Release a few resources to control memory usage.
            original_metadata = upload.pop("metadata", None)
            media = upload.pop("media", None)
            blob.check_preconditions_by_value(
                upload.get("ifGenerationMatch"),
                upload.get("ifGenerationNotMatch"),
                upload.get("ifMetagenerationMatch"),
                upload.get("ifMetagenerationNotMatch"),
            )
            if upload.pop("instructions", None) == "inject-upload-data-error":
                media = testbench_utils.corrupt_media(media)
            revision = blob.insert_resumable(gcs_url, request, media, original_metadata)
            revision.metadata.setdefault("metadata", {})
            revision.metadata["metadata"]["x_testbench_transfer_encoding"] = ":".join(
                upload["transfer"]
            )
            response_payload = testbench_utils.filter_fields_from_response(
                upload.get("fields"), revision.metadata
            )
            upload["payload"] = response_payload
            testbench_utils.insert_object(object_path, blob)

        response = flask.make_response(response_payload)
        if next_byte == 0:
            response.headers["Range"] = "bytes=0-0"
        else:
            response.headers["Range"] = "bytes=0-%d" % (next_byte - 1)
        if upload.get("done", False):
            response.status_code = 200
        else:
            response.status_code = 308
        return response
