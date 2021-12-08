# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Stub Test"""

import os
import sys
import time
import shutil
import datetime
import tempfile
import numpy as np
import pytest

import tensorflow as tf
import tensorflow_io as tfio


def bigtable_func(project_id, instance_id, table_id):
    from google.cloud import bigtable
    from google.cloud.bigtable import column_family
    from google.cloud.bigtable import row_filters
    from google.auth.credentials import AnonymousCredentials

    os.environ["BIGTABLE_EMULATOR_HOST"] = "localhost:8086"

    # [START bigtable_hw_connect]
    # The client must be created with admin=True because it will create a
    # table.
    client = bigtable.Client(
        project=project_id, admin=True, credentials=AnonymousCredentials()
    )
    instance = client.instance(instance_id)
    # [END bigtable_hw_connect]

    # [START bigtable_hw_create_table]
    print(f"Creating the {table_id} table.")
    table = instance.table(table_id)

    print("Creating column family cf1 with Max Version GC rule...")
    # Create a column family with GC policy : most recent N versions
    # Define the GC policy to retain only the most recent 2 versions
    max_versions_rule = column_family.MaxVersionsGCRule(2)
    column_family_id = "cf1"
    column_families = {column_family_id: max_versions_rule}
    if not table.exists():
        table.create(column_families=column_families)
    else:
        print(f"Table {table_id} already exists.")
    # [END bigtable_hw_create_table]

    # [START bigtable_hw_write_rows]
    print("Writing some greetings to the table.")
    greetings = ["Hello World!", "Hello Cloud Bigtable!", "Hello Python!"]
    rows = []
    column = b"greeting"
    for i, value in enumerate(greetings):
        # Note: This example uses sequential numeric IDs for simplicity,
        # but this can result in poor performance in a production
        # application.  Since rows are stored in sorted order by key,
        # sequential keys can result in poor distribution of operations
        # across nodes.
        #
        # For more information about how to design a Bigtable schema for
        # the best performance, see the documentation:
        #
        #     https://cloud.google.com/bigtable/docs/schema-design
        row_key = f"greeting{i}".encode()
        row = table.direct_row(row_key)
        row.set_cell(
            column_family_id, column, value, timestamp=datetime.datetime.utcnow()
        )
        rows.append(row)
    table.mutate_rows(rows)
    # [END bigtable_hw_write_rows]

    # [START bigtable_hw_create_filter]
    # Create a filter to only retrieve the most recent version of the cell
    # for each column accross entire row.
    row_filter = row_filters.CellsColumnLimitFilter(1)
    # [END bigtable_hw_create_filter]

    # [START bigtable_hw_get_with_filter]
    print("Getting a single greeting by row key.")
    key = b"greeting0"

    row = table.read_row(key, row_filter)
    cell = row.cells[column_family_id][column][0]
    print(cell.value.decode("utf-8"))
    # [END bigtable_hw_get_with_filter]

    # [START bigtable_hw_scan_with_filter]
    print("Scanning for all greetings:")
    partial_rows = table.read_rows(filter_=row_filter)

    for row in partial_rows:
        cell = row.cells[column_family_id][column][0]
        print(cell.value.decode("utf-8"))
    # [END bigtable_hw_scan_with_filter]

    # [START bigtable_hw_delete_table]
    print(f"Deleting the {table_id} table.")
    table.delete()
    # [END bigtable_hw_delete_table]


def test_bigtable():
    bigtable_func("bigtable_project", "bigtable_instance", "bigtable_table")
