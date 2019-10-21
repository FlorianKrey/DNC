#!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk
"""Split full rttm file into (sub-)meeting rttm file"""

import argparse

def load_rttm(file_path):
    """ rttm file to dictionary """
    rttm = {}
    with open(file_path, 'r') as _file_path:
        for line in _file_path:
            line = line.split()
            meeting_name = line[1]
            start = float(line[3])
            duration = float(line[4])
            rttm.setdefault(meeting_name, []).append((start, start + duration))
    return rttm

def get_time_boundary(rttm):
    """get time information from rttm into dict"""
    boundaries = {}
    for key, val in rttm.items():
        _, meeting_name, idx = key.split('-')
        earliest = min([i[0] for i in val])
        latest = max([i[1] for i in val])
        boundaries.setdefault(meeting_name, []).append((earliest, latest, idx))
    return boundaries

def segment_rttm(in_file, out_file, boundaries):
    """output (sub-)meeting rttm file"""
    with open(in_file, 'r') as fin, open(out_file, 'w') as fout:
        for line in fin:
            info = line.split()
            meeting_name = info[1].split('-')[1]
            start = float(info[3])
            end = start + float(info[4])
            idx = None
            for section in boundaries[meeting_name]:
                if start >= section[0] and end <= section[1]:
                    idx = section[2]
                    break
            assert idx is not None, "start %f end %f do not fit" %(start, end)
            info[1] += '-%s' % idx
            # hack to make names consistent
            info[1] = info[1].replace('AMIMDM', 'AMI')
            fout.write(' '.join(info) + '\n')

def main():
    """main procedure"""
    parser = argparse.ArgumentParser(description="split rttm files into multiple submeetings")
    parser.add_argument('--submeeting-rttm', required=True, type=str,
                        help="path to an rttm file that is already split into submeetings")
    parser.add_argument('--input-rttm', required=True, type=str,
                        help="path to an input rttm file")
    parser.add_argument('--output-rttm', required=True, type=str,
                        help="path to an output rttm file")
    args = parser.parse_args()

    submeetings = load_rttm(args.submeeting_rttm)
    boundaries = get_time_boundary(submeetings)
    segment_rttm(args.input_rttm, args.output_rttm, boundaries)

if __name__ == '__main__':
    main()
