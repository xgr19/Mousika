#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rule2entry.py
@Time    :   2021/04/27 16:09:47
@Author  :   Guanglin Duan 
@Version :   1.0
convert if-form rule of DT to ternary entry of P4 table
'''
import re
import os

def set_bit_val(byte, index, val):
    """
    set one bit of byte

    :param byte: original byte
    :param index: position
    :param val: target value, 0 or 1
    :returns: modified value
    """
    # 112bits from left to right
    total_bit = 111
    index = total_bit - index
    if val:
        return byte | (1 << index)
    else:
        return byte & ~(1 << index)

class Rule2Binary():
    
    def get_ternary(self, feature_list, tmp_cls):
        """
        @description  :get ternary entry, key & mask = value
        @param        :feature_list
        @Returns      :list [mask, value, cls]
        """
        tmp_mask = 0
        tmp_value = 0
        for feature_index, equal_str, feature_value in feature_list:
            tmp_mask = set_bit_val(tmp_mask, int(feature_index), 1)
            if "!" in equal_str:
                tmp_value = set_bit_val(tmp_value, int(feature_index), int(feature_value)^1)
            else:
                tmp_value = set_bit_val(tmp_value, int(feature_index), int(feature_value))
        return [tmp_mask, tmp_value, tmp_cls]


            

    def parse_file(self, file_path):
        """
        @description  :load rules and get mask and value of every line
        @param        :file path
        @Returns      :list [[mask, value, cls]...]
        """
        
        
        ternary_list = []   
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                pattern = re.compile(r'feature_(\d*)(\D*=)(\d)')
                matchObj = pattern.findall(line)
                tmp_cls = int(line.split()[-1])
                ternary_list.append(self.get_ternary(matchObj, tmp_cls))
        return ternary_list
    
    def write_ternary_file(self, ternary_list, dec_file_path):
        """
        @description  :output mask,value,cls to file
        @param        :ternary_list
        @Returns      :output files in decimal formats
        """
        with open(dec_file_path, 'w', encoding='utf-8') as f:
            for data in ternary_list:
                f.write("{} {} {}\n".format(*data))
        f.close()
        

    def load_ternary(self, dec_file_path):
        """
        @description  :load ternary entry
        @param        :
        @Returns      :list [[mask, value, cls]...]
        """
        ternary_list = []
        with open(dec_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                ternary_list.append([int(x) for x in line.strip().split()])
        return ternary_list

        

def file_name_walk(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".txt":
                file_list.append("{}/{}".format(root, file))
    return file_list


def rule2entry():
    root_path = '../output/rule_tree/univ'
    save_root = '../output/ternary_entry/univ'
    file_list = file_name_walk(root_path)
    for input_file in file_list:
        file_name = input_file.split('/')[-1].split('.')[0]
        print(file_name)
        save_file = '{}/{}.txt'.format(save_root, file_name)
        rule2binaly = Rule2Binary()
        ternary_list = rule2binaly.parse_file(input_file)
        rule2binaly.write_ternary_file(ternary_list, save_file)

def main():
    rule2entry()

if __name__ == "__main__":
    main()