# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from io import BytesIO
from avro.io import DatumReader, DatumWriter, BinaryDecoder, BinaryEncoder
from avro.datafile import DataFileReader, DataFileWriter
from avro.schema import Parse as parse


class AvroRecordsToFile(object):
    def __init__(self, filename, writer_schema, codec='deflate'):
        """

        :param filename:
        :param writer_schema:
        :param codec:
        """
        self.schema = AvroParser(writer_schema).get_schema_object()
        self.filename = filename
        self.codec = codec

    def write_records(self, records):
        with open(self.filename, 'wb') as out:
            writer = DataFileWriter(
                out, DatumWriter(), self.schema, codec=self.codec)
            for record in records:
                writer.append(record)
            writer.close()


class AvroFileToRecords(object):
    def __init__(self, filename, reader_schema=None):
        """
        Reads records as strings where each row is serialized separately

        :param filename: The filename from where to load the records
        :param reader_schema: Schema used for reading

        :return: An array of serialized string with one string per record
        """
        self.records = []

        with open(filename, 'rb') as file_handle:
            datum_reader = DatumReader(reader_schema=AvroParser(reader_schema).get_schema_object()) \
                if reader_schema else DatumReader()
            reader = DataFileReader(file_handle, datum_reader)

            self.records += [record for record in reader]

    def get_records(self):
        return self.records


class AvroSchemaReader(object):
    def __init__(self, filename):
        """
        Reads the schema from a file into json string
        """
        with open(filename, 'rb') as file_handle:
            reader = DataFileReader(file_handle, DatumReader())
            self.schema_json = ""
            if six.PY2:
                self.schema_json = str(reader.datum_reader.writers_schema)

            elif six.PY3:
                self.schema_json = str(reader.datum_reader.writer_schema)

            else:
                raise RuntimeError("Only python 2 and python 3 are supported!")

    def get_schema_json(self):
        return self.schema_json


class AvroParser(object):

    def __init__(self, schema_json):
        """
        Create an avro parser mostly to abstract away the API change between
        avro and avro-python3

        :param schema_json:
        """
        self.schema_object = parse(schema_json)

    def get_schema_object(self):
        return self.schema_object


class AvroDeserializer(object):

    def __init__(self, schema_json):
        """
        Create an avro deserializer.

        :param schema_json: Json string of the schema.
        """
        schema_object = AvroParser(schema_json).get_schema_object()
        # No schema resolution
        self.datum_reader = DatumReader(schema_object, schema_object)

    def deserialize(self, serialized_bytes):
        """
        Deserialize an avro record from bytes.

        :param serialized_bytes: The serialized bytes input.

        :return: The de-serialized record structure in python as map-list object.
        """
        return self.datum_reader.read(BinaryDecoder(BytesIO(serialized_bytes)))


class AvroSerializer(object):

    def __init__(self, schema_json):
        """
        Create an avro serializer.

        :param schema_json: Json string of the schema.
        """
        self.datum_writer = DatumWriter(
            AvroParser(schema_json).get_schema_object())

    def serialize(self, datum):
        """
        Serialize a datum into a avro formatted string.

        :param datum: The avro datum.

        :return: The serialized bytes.
        """
        writer = BytesIO()
        self.datum_writer.write(datum, BinaryEncoder(writer))
        return writer.getvalue()
