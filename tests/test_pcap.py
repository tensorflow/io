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
This test expands on the DPKT HTTP example and the TF IO test_text code.
https://github.com/deeprtc/dpkt/blob/master/examples/print_http_requests.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from tensorflow import errors  # pylint: disable=wrong-import-position
import tensorflow_io.text as text_io  # pylint: disable=wrong-import-position

import dpkt
import datetime
import socket
from dpkt.compat import compat_ord


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def print_http_requests(pcap):
    """Print out information about each packet in a pcap

       Args:
           pcap: dpkt pcap reader object (dpkt.pcap.Reader)
    """
    # For each packet in the pcap process the contents
    for timestamp, buf in pcap:

        # Unpack the Ethernet frame (mac src/dst, ethertype)
        eth = dpkt.ethernet.Ethernet(buf)

        # Make sure the Ethernet data contains an IP packet
        if not isinstance(eth.data, dpkt.ip.IP):
            print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue

        # Now grab the data within the Ethernet frame (the IP packet)
        ip = eth.data

        # Check for TCP in the transport layer
        if isinstance(ip.data, dpkt.tcp.TCP):

            # Set the TCP data
            tcp = ip.data

            # Now see if we can parse the contents as a HTTP request
            try:
                request = dpkt.http.Request(tcp.data)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

            # Pull out fragment information (flags and offset all packed into off field, so use bitmasks)
            do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
            more_fragments = bool(ip.off & dpkt.ip.IP_MF)
            fragment_offset = ip.off & dpkt.ip.IP_OFFMASK

            # Print out the info
            print('Timestamp: ', str(datetime.datetime.utcfromtimestamp(timestamp)))
            print('Ethernet Frame: ', mac_addr(eth.src), mac_addr(eth.dst), eth.type)
            print('IP: %s -> %s   (len=%d ttl=%d DF=%d MF=%d offset=%d)' %
                  (inet_to_str(ip.src), inet_to_str(ip.dst), ip.len, ip.ttl, do_not_fragment, more_fragments,
                   fragment_offset))
            print('HTTP request: %s\n' % repr(request))

            # Check for Header spanning acrossed TCP segments
            if not tcp.data.endswith(b'\r\n'):
                print('\nHEADER TRUNCATED! Reassemble TCP segments!\n')


def test_http_pcap():
    """Open up a test pcap file and print out the packets"""
    with open('test_pcap/http.pcap', 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        print_http_requests(pcap)


def test_pcap_input():
    """test_pcap_input
    """
    pcap_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_pcap", "http.pcap")

    with open(pcap_filename, 'rb') as f:
        pcapr = dpkt.pcap.Reader(f)
        # read packets from test pcap file in a memory array
        pcap = [p for p in pcapr]

    pcap = pcap*2

    file_url = "file://" + pcap_filename
    url_filenames = [file_url, file_url]
    dataset = text_io.PcapDataset(url_filenames, batch=2)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        for i in range(0, len(pcap), 2)
            v = sess.run(get_next)
            for j in range(2)
                orig_timestamp, orig_buf = pcap[i+j]
                sample_timestamp, sample_buf = v[j]
                assert orig_timestamp == sample_timestamp
                assert orig_buf == sample_buf
        with pytest.raises(errors.OutOfRangeError):
            sess.run(get_next)


if __name__ == "__main__":
    test.main()
