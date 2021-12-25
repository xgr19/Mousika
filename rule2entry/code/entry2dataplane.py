#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rule2entry.py
@Time    :   2021-01-08 16:46:41
@Author  :   Guanglin Duan 
@Version :   1.0
load ternary entry and send to P4 dataplane
'''
import logging
import random

from ptf import config
import ptf.testutils as testutils
from bfruntime_client_base_tests import BfRuntimeTest
import bfrt_grpc.bfruntime_pb2 as bfruntime_pb2
import bfrt_grpc.client as gc

##### ******************* #####

logger = logging.getLogger('Test')
if not len(logger.handlers):
    logger.addHandler(logging.StreamHandler())

swports = []
for device, port, ifname in config["interfaces"]:
    swports.append(port)
    swports.sort()

if swports == []:
    swports = list(range(9))

# from cls to mac
def cls2mac(tmp_cls):
    return "00:00:00:02:02:0{}".format(tmp_cls)

# from mac str to int
def mac2int(mac_str):
    return int(mac_str.replace(':', ''), 16)

data_type = ['iot-attack-retest', 'univ', 'iscx']
data_type_index = 2
file_name_list = ['univ_gru.txt', 'univ_lstm.txt', 'univ_rf.txt']
file_name_index = 0
total_entry_number = 1

class DecisionTreeTest(BfRuntimeTest):
    def setUp(self):
        client_id = 0
        p4_name = "flowcontrol"
        BfRuntimeTest.setUp(self, client_id, p4_name)
    def get_entry(self):
        """
        @description  :load ternary entry from file
        @param        :
        @Returns      :list [[mask, value, cls]...]
        """
        ternary_list = []
        
        dec_file_path = "'../output/ternary_entry/{}/{}".format(data_type[data_type_index], file_name_list[file_name_index])
        with open(dec_file_path, 'r') as f:
            for line in f:
                ternary_list.append([int(x) for x in line.strip().split()])
        return ternary_list

    def my_add_table_entry(self):
        p4_name = "flowcontrol"

        # Get bfrt_info and set it as part of the test
        bfrt_info = self.interface.bfrt_info_get(p4_name)
        
        target = gc.Target(device_id=0, pipe_id=0xffff)

        # get table entry
        ternary_list = self.get_entry()

        # table tb_packet_cls
        logger.info("Insert tb_packet_cls table entry")
        tb_packet_cls = bfrt_info.table_get("Ingress.tb_packet_cls")
        entry_count = 0
        for tmp_mask, tmp_value, tmp_cls in ternary_list:
            print(tmp_mask, tmp_value, tmp_cls)
            tb_packet_cls.entry_add(
                target,
                [tb_packet_cls.make_key(
                    #('key', value, mask)
                    [gc.KeyTuple('meta.bin_feature', int(tmp_value), int(tmp_mask)),
                     gc.KeyTuple('$MATCH_PRIORITY', 1)]
                    # [gc.KeyTuple('meta.bin_feature', int(tmp_mask), int(tmp_value))]
                )],
                [tb_packet_cls.make_data(
                    [gc.DataTuple('dstAddr', mac2int(cls2mac(tmp_cls))),
                    gc.DataTuple('port', 0)], 
                    'Ingress.ac_packet_forward')]
                
            )
            entry_count += 1

        logger.info("add entry ok")
        logger.info(file_name_list[file_name_index])
        logger.info(entry_count)


    def runTest(self):
        self.my_add_table_entry()