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
"""Implement a class to simulate GCS objects."""

import base64
import crc32c
import error_response
import hashlib
import json
import struct
import testbench_utils
import time


class GcsObjectVersion:
    """Represent a single revision of a GCS Object."""

    def __init__(self, gcs_url, bucket_name, name, generation, request, media):
        """Initialize a new object revision.

        :param gcs_url:str the base URL for the GCS service.
        :param bucket_name:str the name of the bucket that contains the object.
        :param name:str the name of the object.
        :param generation:int the generation number for this object.
        :param request:flask.Request the contents of the HTTP request.
        :param media:str the contents of the object.
        """
        self.gcs_url = gcs_url
        self.bucket_name = bucket_name
        self.name = name
        self.generation = str(generation)
        self.object_id = bucket_name + "/o/" + name + "/" + str(generation)
        now = time.gmtime(time.time())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", now)
        self.media = media
        instructions = request.headers.get("x-goog-testbench-instructions")
        if instructions == "inject-upload-data-error":
            self.media = testbench_utils.corrupt_media(media)

        self.metadata = {
            "timeCreated": timestamp,
            "updated": timestamp,
            "metageneration": "0",
            "generation": str(generation),
            "location": "US",
            "storageClass": "STANDARD",
            "size": str(len(self.media)),
            "etag": "XYZ=",
            "owner": {"entity": "project-owners-123456789", "entityId": ""},
            "md5Hash": base64.b64encode(hashlib.md5(self.media).digest()).decode(
                "utf-8"
            ),
            "crc32c": base64.b64encode(
                struct.pack(">I", crc32c.crc32(self.media))
            ).decode("utf-8"),
        }
        if request.headers.get("content-type") is not None:
            self.metadata["contentType"] = request.headers.get("content-type")

    def update_from_metadata(self, metadata):
        """Update from a metadata dictionary.

        :param metadata:dict a dictionary with new metadata values.
        :rtype:NoneType
        """
        tmp = self.metadata.copy()
        tmp.update(metadata)
        tmp["bucket"] = tmp.get("bucket", self.name)
        tmp["name"] = tmp.get("name", self.name)
        now = time.gmtime(time.time())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", now)
        # Some values cannot be changed via updates, so we always reset them.
        tmp.update(
            {
                "kind": "storage#object",
                "bucket": self.bucket_name,
                "name": self.name,
                "id": self.object_id,
                "selfLink": self.gcs_url + self.name,
                "projectNumber": "123456789",
                "updated": timestamp,
            }
        )
        tmp["metageneration"] = str(int(tmp.get("metageneration", "0")) + 1)
        self.metadata = tmp
        self._validate_hashes()

    def _validate_hashes(self):
        """Validate the md5Hash and crc32c fields against the stored media."""
        self._validate_md5_hash()
        self._validate_crc32c()

    def _validate_md5_hash(self):
        """Validate the md5Hash field against the stored media."""
        actual = self.metadata.get("md5Hash", "")
        expected = base64.b64encode(hashlib.md5(self.media).digest()).decode("utf-8")
        if actual != expected:
            raise error_response.ErrorResponse(
                f"Mismatched MD5 hash expected={expected}, actual={actual}"
            )

    def _validate_crc32c(self):
        """Validate the crc32c field against the stored media."""
        actual = self.metadata.get("crc32c", "")
        expected = base64.b64encode(struct.pack(">I", crc32c.crc32(self.media))).decode(
            "utf-8"
        )
        if actual != expected:
            raise error_response.ErrorResponse(
                "Mismatched CRC32C checksum expected={}, actual={}".format(
                    expected, actual
                )
            )

    def validate_encryption_for_read(self, request, prefix="x-goog-encryption"):
        """Verify that the request includes the correct encryption keys.

        :param request:flask.Request the http request.
        :param prefix: str the prefix shared by the encryption headers,
            typically 'x-goog-encryption', but for rewrite requests it can be
            'x-goog-copy-source-encryption'.
        :rtype:NoneType
        """
        key_header = prefix + "-key"
        hash_header = prefix + "-key-sha256"
        algo_header = prefix + "-algorithm"
        encryption = self.metadata.get("customerEncryption")
        if encryption is None:
            # The object is not encrypted, no key is needed.
            if request.headers.get(key_header) is None:
                return
            else:
                # The data is not encrypted, sending an encryption key is an
                # error.
                testbench_utils.raise_csek_error()
        # The data is encrypted, the key must be present, match, and match its
        # hash.
        key_header_value = request.headers.get(key_header)
        hash_header_value = request.headers.get(hash_header)
        algo_header_value = request.headers.get(algo_header)
        testbench_utils.validate_customer_encryption_headers(
            key_header_value, hash_header_value, algo_header_value
        )
        if encryption.get("keySha256") != hash_header_value:
            testbench_utils.raise_csek_error()

    def _capture_customer_encryption(self, request):
        """Capture the customer-supplied encryption key, if any.

        :param request:flask.Request the http request.
        :rtype:NoneType
        """
        if request.headers.get("x-goog-encryption-key") is None:
            return
        prefix = "x-goog-encryption"
        key_header = prefix + "-key"
        hash_header = prefix + "-key-sha256"
        algo_header = prefix + "-algorithm"
        key_header_value = request.headers.get(key_header)
        hash_header_value = request.headers.get(hash_header)
        algo_header_value = request.headers.get(algo_header)
        testbench_utils.validate_customer_encryption_headers(
            key_header_value, hash_header_value, algo_header_value
        )
        self.metadata["customerEncryption"] = {
            "encryptionAlgorithm": algo_header_value,
            "keySha256": hash_header_value,
        }

    def x_goog_hash_header(self):
        """Return the value for the x-goog-hash header."""
        hashes = {
            "md5": self.metadata.get("md5Hash", ""),
            "crc32c": self.metadata.get("crc32c", ""),
        }
        hashes = [f"{key}={val}" for key, val in hashes.items() if val]
        return ",".join(hashes)


class GcsObject:
    """Represent a GCS Object, including all its revisions."""

    def __init__(self, bucket_name, name):
        """Initialize a fake GCS Blob.

        :param bucket_name:str the bucket that will contain the new object.
        :param name:str the name of the new object.
        """
        self.bucket_name = bucket_name
        self.name = name
        # A counter to create new generation numbers for the object revisions.
        # Note that 0 is an invalid generation number. The application can use
        # ifGenerationMatch=0 as a pre-condition that means "object does not
        # exist".
        self.generation_generator = 0
        self.current_generation = None
        self.revisions = {}
        self.rewrite_token_generator = 0
        self.rewrite_operations = {}

    def get_revision(self, request, version_field_name="generation"):
        """Get the information about a particular object revision or raise.

        :param request:flask.Request the contents of the http request.
        :param version_field_name:str the name of the generation
            parameter, typically 'generation', but sometimes 'sourceGeneration'.
        :return: the object revision.
        :rtype: GcsObjectVersion
        :raises:ErrorResponse if the request contains an invalid generation
            number.
        """
        generation = request.args.get(version_field_name)
        if generation is None:
            return self.get_latest()
        version = self.revisions.get(generation)
        if version is None:
            raise error_response.ErrorResponse(
                "Precondition Failed: generation %s not found" % generation
            )
        return version

    def del_revision(self, request):
        """Delete a version of a fake GCS Blob.

        :param request:flask.Request the contents of the HTTP request.
        :return: True if the object entry in the Bucket should be deleted.
        :rtype: bool
        """
        generation = request.args.get("generation") or self.current_generation
        if generation is None:
            return True
        self.revisions.pop(generation)
        if len(self.revisions) == 0:
            self.current_generation = None
            return True
        self.current_generation = sorted(self.revisions.keys())[-1]
        return False

    @classmethod
    def _remove_non_writable_keys(cls, metadata):
        """Remove the keys from metadata (an update or patch) that are not
        writable.

        Both `Objects: patch` and `Objects: update` either ignore non-writable
        keys or return 400 if the key does not match the current value. In
        the testbench we simply always ignore them, to make life easier.

        :param metadata:dict a dictionary representing a patch or
            update to the metadata.
        :return metadata but with only any non-writable keys removed.
        :rtype: dict
        """
        writeable_keys = {
            "acl",
            "cacheControl",
            "contentDisposition",
            "contentEncoding",
            "contentLanguage",
            "contentType",
            "eventBasedHold",
            "metadata",
            "temporaryHold",
            "storageClass",
            "customTime",
        }
        # Cannot change `metadata` while we are iterating over it, so we make
        # a copy
        keys = [key for key in metadata.keys()]
        for key in keys:
            if key not in writeable_keys:
                metadata.pop(key, None)
        return metadata

    def get_revision_by_generation(self, generation):
        """Get object revision by generation or None if not found.

        :param generation:int
        :return: the object revision by generation or None.
        :rtype:GcsObjectRevision
        """
        return self.revisions.get(str(generation), None)

    def get_latest(self):
        return self.revisions.get(self.current_generation, None)

    def check_preconditions_by_value(
        self,
        generation_match,
        generation_not_match,
        metageneration_match,
        metageneration_not_match,
    ):
        """Verify that the given precondition values are met."""
        current_generation = self.current_generation or "0"
        if generation_match is not None and generation_match != current_generation:
            raise error_response.ErrorResponse("Precondition Failed", status_code=412)
        # This object does not exist (yet), testing in this case is special.
        if (
            generation_not_match is not None
            and generation_not_match == current_generation
        ):
            raise error_response.ErrorResponse("Precondition Failed", status_code=412)

        if self.current_generation is None:
            if metageneration_match is not None or metageneration_not_match is not None:
                raise error_response.ErrorResponse(
                    "Precondition Failed", status_code=412
                )
            return

        current = self.revisions.get(current_generation)
        if current is None:
            raise error_response.ErrorResponse("Object not found", status_code=404)
        metageneration = current.metadata.get("metageneration")
        if (
            metageneration_not_match is not None
            and metageneration_not_match == metageneration
        ):
            raise error_response.ErrorResponse("Precondition Failed", status_code=412)
        if metageneration_match is not None and metageneration_match != metageneration:
            raise error_response.ErrorResponse("Precondition Failed", status_code=412)

    def check_preconditions(
        self,
        request,
        if_generation_match="ifGenerationMatch",
        if_generation_not_match="ifGenerationNotMatch",
        if_metageneration_match="ifMetagenerationMatch",
        if_metageneration_not_match="ifMetagenerationNotMatch",
    ):
        """Verify that the preconditions in request are met.

        :param request:flask.Request the http request.
        :param if_generation_match:str the name of the generation match
            parameter name, typically 'ifGenerationMatch', but sometimes
            'ifSourceGenerationMatch'.
        :param if_generation_not_match:str the name of the generation not-match
            parameter name, typically 'ifGenerationNotMatch', but sometimes
            'ifSourceGenerationNotMatch'.
        :param if_metageneration_match:str the name of the metageneration match
            parameter name, typically 'ifMetagenerationMatch', but sometimes
            'ifSourceMetagenerationMatch'.
        :param if_metageneration_not_match:str the name of the metageneration
            not-match parameter name, typically 'ifMetagenerationNotMatch', but
            sometimes 'ifSourceMetagenerationNotMatch'.
        :rtype:NoneType
        """
        generation_match = request.args.get(if_generation_match)
        generation_not_match = request.args.get(if_generation_not_match)
        metageneration_match = request.args.get(if_metageneration_match)
        metageneration_not_match = request.args.get(if_metageneration_not_match)
        self.check_preconditions_by_value(
            generation_match,
            generation_not_match,
            metageneration_match,
            metageneration_not_match,
        )

    def _insert_revision(self, revision):
        """Insert a new revision that has been initialized and checked.

        :param revision: GcsObjectVersion the new revision to insert.
        :rtype:NoneType
        """
        update = {str(self.generation_generator): revision}
        bucket = testbench_utils.lookup_bucket(self.bucket_name)
        if not bucket.versioning_enabled():
            self.revisions = update
        else:
            self.revisions.update(update)
        self.current_generation = str(self.generation_generator)

    def insert(self, gcs_url, request):
        """Insert a new revision based on the give flask request.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :return: the newly created object version.
        :rtype: GcsObjectVersion
        """
        media = testbench_utils.extract_media(request)
        self.generation_generator += 1
        revision = GcsObjectVersion(
            gcs_url,
            self.bucket_name,
            self.name,
            self.generation_generator,
            request,
            media,
        )
        meta = revision.metadata.setdefault("metadata", {})
        meta["x_testbench_upload"] = "simple"
        self._insert_revision(revision)
        return revision

    def insert_multipart(self, gcs_url, request, resource, media_headers, media_body):
        """Insert a new revision based on the give flask request.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :param resource:dict JSON resource with object metadata.
        :param media_headers:dict media headers in a multi-part upload.
        :param media_body:str object data in a multi-part upload.
        :return: the newly created object version.
        :rtype: GcsObjectVersion
        """
        # There are two ways to specify the content-type, the 'content-type'
        # header and the resource['contentType'] field. They must be consistent,
        # and the service generates an error when they are not.
        if (
            resource.get("contentType") is not None
            and media_headers.get("content-type") is not None
            and resource.get("contentType") != media_headers.get("content-type")
        ):
            raise error_response.ErrorResponse(
                (
                    "Content-Type specified in the upload (%s) does not match"
                    + "contentType specified in the metadata (%s)."
                )
                % (media_headers.get("content-type"), resource.get("contentType")),
                status_code=400,
            )
        # Set the contentType in the resource from the header. Note that if both
        # are set they have the same value.
        resource.setdefault("contentType", media_headers.get("content-type"))
        self.generation_generator += 1
        revision = GcsObjectVersion(
            gcs_url,
            self.bucket_name,
            self.name,
            self.generation_generator,
            request,
            media_body,
        )
        meta = revision.metadata.setdefault("metadata", {})
        meta["x_testbench_upload"] = "multipart"
        if "md5Hash" in resource:
            # We should return `x_testbench_md5` only when the user enables
            # `MD5Hash` computations.
            meta["x_testbench_md5"] = resource.get("md5Hash")
        meta["x_testbench_crc32c"] = resource.get("crc32c", "")
        # Apply any overrides from the resource object part.
        revision.update_from_metadata(resource)
        self._insert_revision(revision)
        return revision

    def insert_resumable(self, gcs_url, request, media, resource):
        """Implement the final insert for a resumable upload.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :param media:str the media for the object.
        :param resource:dict the metadata for the object.
        :return: the newly created object version.
        :rtype: GcsObjectVersion
        """
        self.generation_generator += 1
        revision = GcsObjectVersion(
            gcs_url,
            self.bucket_name,
            self.name,
            self.generation_generator,
            request,
            media,
        )
        meta = revision.metadata.setdefault("metadata", {})
        meta["x_testbench_upload"] = "resumable"
        meta["x_testbench_md5"] = resource.get("md5Hash", "")
        meta["x_testbench_crc32c"] = resource.get("crc32c", "")
        # Apply any overrides from the resource object part.
        revision.update_from_metadata(resource)
        self._insert_revision(revision)
        return revision

    def insert_xml(self, gcs_url, request):
        """Implement the insert operation using the XML API.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :return: the newly created object version.
        :rtype: GcsObjectVersion
        """
        media = testbench_utils.extract_media(request)
        self.generation_generator += 1
        goog_hash = request.headers.get("x-goog-hash")
        md5hash = None
        crc32c = None
        if goog_hash is not None:
            for hash in goog_hash.split(","):
                if hash.startswith("md5="):
                    md5hash = hash[4:]
                if hash.startswith("crc32c="):
                    crc32c = hash[7:]
        revision = GcsObjectVersion(
            gcs_url,
            self.bucket_name,
            self.name,
            self.generation_generator,
            request,
            media,
        )
        meta = revision.metadata.setdefault("metadata", {})
        meta["x_testbench_upload"] = "xml"
        if md5hash is not None:
            meta["x_testbench_md5"] = md5hash
            revision.update_from_metadata({"md5Hash": md5hash})
        if crc32c is not None:
            meta["x_testbench_crc32c"] = crc32c
            revision.update_from_metadata({"crc32c": crc32c})
        self._insert_revision(revision)
        return revision

    def copy_from(self, gcs_url, request, source_revision):
        """Insert a new revision based on the give flask request.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :param source_revision:GcsObjectVersion the source object version to
            copy from.
        :return: the newly created object version.
        :rtype: GcsObjectVersion
        """
        self.generation_generator += 1
        source_revision.validate_encryption_for_read(request)
        revision = GcsObjectVersion(
            gcs_url,
            self.bucket_name,
            self.name,
            self.generation_generator,
            request,
            source_revision.media,
        )
        metadata = json.loads(request.data)
        revision.update_from_metadata(metadata)
        self._insert_revision(revision)
        return revision

    def compose_from(self, gcs_url, request, composed_media):
        """Compose a new revision based on the give flask request.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :param composed_media:str contents of the composed object
        :return: the newly created object version.
        :rtype: GcsObjectVersion
        """
        self.generation_generator += 1
        revision = GcsObjectVersion(
            gcs_url,
            self.bucket_name,
            self.name,
            self.generation_generator,
            request,
            composed_media,
        )
        payload = json.loads(request.data)
        if payload.get("destination") is not None:
            revision.update_from_metadata(payload.get("destination"))
        # The server often discards the MD5 Hash when composing objects, we can
        # easily maintain them in the testbench, but dropping them helps us
        # detect bugs sooner.
        revision.metadata.pop("md5Hash")
        self._insert_revision(revision)
        return revision

    @classmethod
    def rewrite_fixed_args(cls):
        """The arguments that should not change between requests for the same
        rewrite operation."""
        return [
            "destinationKmsKeyName",
            "destinationPredefinedAcl",
            "ifGenerationMatch",
            "ifGenerationNotMatch",
            "ifMetagenerationMatch",
            "ifMetagenerationNotMatch",
            "ifSourceGenerationMatch",
            "ifSourceGenerationNotMatch",
            "ifSourceMetagenerationMatch",
            "ifSourceMetagenerationNotMatch",
            "maxBytesRewrittenPerCall",
            "projection",
            "sourceGeneration",
            "userProject",
        ]

    @classmethod
    def capture_rewrite_operation_arguments(
        cls, request, destination_bucket, destination_object
    ):
        """Captures the arguments used to validate related rewrite calls.

        :rtype:dict
        """
        original_arguments = {}
        for arg in GcsObject.rewrite_fixed_args():
            original_arguments[arg] = request.args.get(arg)
        original_arguments.update(
            {
                "destination_bucket": destination_bucket,
                "destination_object": destination_object,
            }
        )
        return original_arguments

    @classmethod
    def make_rewrite_token(
        cls, operation, destination_bucket, destination_object, generation
    ):
        """Create a new rewrite token for the given operation."""
        return base64.b64encode(
            bytearray(
                "/".join(
                    [
                        str(operation.get("id")),
                        destination_bucket,
                        destination_object,
                        str(generation),
                        str(operation.get("bytes_rewritten")),
                    ]
                ),
                "utf-8",
            )
        ).decode("utf-8")

    def make_rewrite_operation(self, request, destination_bucket, destination_object):
        """Create a new rewrite token for `Objects: rewrite`."""
        generation = request.args.get("sourceGeneration")
        if generation is None:
            generation = str(self.generation_generator)
        else:
            generation = generation

        self.rewrite_token_generator = self.rewrite_token_generator + 1
        body = json.loads(request.data)
        original_arguments = self.capture_rewrite_operation_arguments(
            request, destination_object, destination_object
        )
        operation = {
            "id": self.rewrite_token_generator,
            "original_arguments": original_arguments,
            "actual_generation": generation,
            "bytes_rewritten": 0,
            "body": body,
        }
        token = GcsObject.make_rewrite_token(
            operation, destination_bucket, destination_object, generation
        )
        return token, operation

    def rewrite_finish(self, gcs_url, request, body, source):
        """Complete a rewrite from `source` into this object.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :param body:dict the HTTP payload, parsed via json.loads()
        :param source:GcsObjectVersion the source object version.
        :return: the newly created object version.
        :rtype: GcsObjectVersion
        """
        media = source.media
        self.check_preconditions(request)
        self.generation_generator += 1
        revision = GcsObjectVersion(
            gcs_url,
            self.bucket_name,
            self.name,
            self.generation_generator,
            request,
            media,
        )
        revision.update_from_metadata(body)
        self._insert_revision(revision)
        return revision

    def rewrite_step(self, gcs_url, request, destination_bucket, destination_object):
        """Execute an iteration of `Objects: rewrite.

        Objects: rewrite may need to be called multiple times before it
        succeeds. Only objects in the same location, with the same encryption,
        are guaranteed to complete in a single request.

        The implementation simulates some, but not all, the behaviors of the
        server, in particular, only rewrites within the same bucket and smaller
        than 1MiB complete immediately.

        :param gcs_url:str the root URL for the fake GCS service.
        :param request:flask.Request the contents of the HTTP request.
        :param destination_bucket:str where will the object be placed after the
            rewrite operation completes.
        :param destination_object:str the name of the object when the rewrite
            operation completes.
        :return: a dictionary prepared for JSON encoding of a
            `Objects: rewrite` response.
        :rtype:dict
        """
        body = json.loads(request.data)
        rewrite_token = request.args.get("rewriteToken")
        if rewrite_token is not None and rewrite_token != "":
            # Note that we remove the rewrite operation, not just look it up.
            # That way if the operation completes in this call, and/or fails,
            # it is already removed. We need to insert it with a new token
            # anyway, so this makes sense.
            rewrite = self.rewrite_operations.pop(rewrite_token, None)
            if rewrite is None:
                raise error_response.ErrorResponse(
                    "Invalid or expired token in rewrite", status_code=410
                )
        else:
            rewrite_token, rewrite = self.make_rewrite_operation(
                request, destination_bucket, destination_bucket
            )

        # Compare the difference to the original arguments, on the first call
        # this is a waste, but the code is easier to follow.
        current_arguments = self.capture_rewrite_operation_arguments(
            request, destination_bucket, destination_object
        )
        diff = set(current_arguments) ^ set(rewrite.get("original_arguments"))
        if len(diff) != 0:
            raise error_response.ErrorResponse(
                "Mismatched arguments to rewrite", status_code=412
            )

        # This will raise if the version is deleted while the operation is in
        # progress.
        source = self.get_revision_by_generation(rewrite.get("actual_generation"))
        source.validate_encryption_for_read(
            request, prefix="x-goog-copy-source-encryption"
        )
        bytes_rewritten = rewrite.get("bytes_rewritten")
        bytes_rewritten += 1024 * 1024
        result = {"kind": "storage#rewriteResponse", "objectSize": len(source.media)}
        if bytes_rewritten >= len(source.media):
            bytes_rewritten = len(source.media)
            rewrite["bytes_rewritten"] = bytes_rewritten
            # Success, the operation completed. Return the new object:
            object_path, destination = testbench_utils.get_object(
                destination_bucket,
                destination_object,
                GcsObject(destination_bucket, destination_object),
            )
            revision = destination.rewrite_finish(gcs_url, request, body, source)
            testbench_utils.insert_object(object_path, destination)
            result["done"] = True
            result["resource"] = revision.metadata
            rewrite_token = ""
        else:
            rewrite["bytes_rewritten"] = bytes_rewritten
            rewrite_token = GcsObject.make_rewrite_token(
                rewrite, destination_bucket, destination_object, source.generation
            )
            self.rewrite_operations[rewrite_token] = rewrite
            result["done"] = False

        result.update(
            {"totalBytesRewritten": bytes_rewritten, "rewriteToken": rewrite_token}
        )
        return result
