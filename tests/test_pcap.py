# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Test PcapDataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import tensorflow_io.pcap as pcap_io # pylint: disable=wrong-import-position

def test_pcap_input():
    """test_pcap_input
    """
    print("Testing PcapDataset")
    pcap_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_pcap", "http.pcap")
    file_url = "file://" + pcap_filename
    url_filenames = [file_url]
    dataset = pcap_io.PcapDataset(url_filenames, batch=1)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        try:
            print("Reading pcap data via tf session")
            v = sess.run(get_next)
            # sample_timestamp, sample_buf = v[0]
            # assert sample_timestamp != None
            # assert sample_buf != None
        except tf.errors.OutOfRangeError as err:
            print("Exception caught during test:")
            print(err)
            # break


if __name__ == "__main__":
    test.main()
