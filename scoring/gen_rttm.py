#!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk
"""Generate RTTM file from json output"""
import os
import sys
import argparse
import json
from collections import deque

import numpy as np
import kaldiio
import utils

def setup():
    """Get cmds and setup directories"""
    parser = argparse.ArgumentParser(description="Generate RTTM file from json output")
    parser.add_argument('--input-scp', type=str, required=True, action=utils.Abspath,
                        help="the scp file of the input data")
    parser.add_argument('--js-dir', type=str, required=True, action=utils.Abspath,
                        help="the json file contains the clustering result")
    parser.add_argument('--js-num', type=int, required=True,
                        help="the number of json files to load")
    parser.add_argument('--js-name', type=str, default='data',
                        help='the name of the json file name.num.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help="the output rttm file for scoring, default is the json dir")
    parser.add_argument('--rttm-name', type=str, default='output',
                        help="the rttm name, gives output_dir/xxx.rttm")
    parser.add_argument('--min-len', type=int, default=1,
                        help='the minimun length of the dvector to be included')
    args = parser.parse_args()

    if not args.input_scp.endswith('.scp'):
        utils.print_error("scp path has to end with .scp")
    if args.output_dir is None:
        args.output_dir = args.js_dir
    else:
        args.output_dir = utils.get_abs_path(args.output_dir)
    utils.check_output_dir(args.output_dir, True)
    utils.cache_command(sys.argv, args.output_dir)
    utils.change_dir(args.output_dir)

    return args

def load_scp(scp_file):
    """load scp file to get meetingnames and their length"""
    meetings = {}
    with open(scp_file, 'r') as _scp_file:
        for line in _scp_file:
            segment_name, ark_pointer = line.split()
            meeting_name = 'AMI-' + segment_name.split('-')[2]
            start_time = int(segment_name.split('_')[2])
            end_time = int(segment_name.split('_')[3])
            num_frames = kaldiio.load_mat(ark_pointer).shape[0]
            meetings.setdefault(meeting_name, []).append((start_time, end_time, num_frames))
    return meetings

def filter_encompassed_segments(segments, min_length):
    """Remove segments completely contained within another"""
    sorted_segments = sorted(segments, key=lambda x: x[0])
    encompassed_indicator = []
    for segment in segments:
        start, end, length = segment
        start_before = [i for i in sorted_segments if i[0] <= start]
        end_after = [i for i in sorted_segments if i[1] >= end]
        start_before.remove(segment)
        end_after.remove(segment)
        cur_indicator = False
        if not set(start_before).isdisjoint(end_after):
            cur_indicator = True
        if length < min_length:
            cur_indicator = True
        encompassed_indicator.append(cur_indicator)
    return encompassed_indicator

def write_rttm(meetings, file_path):
    """write the rttm file"""
    with open(file_path, 'w') as _rttm_file:
        for meeting, segments in meetings.items():
            for segment in segments:
                start = segment[0] / 100
                duration = (segment[1] - segment[0]) / 100
                _rttm_file.write('SPEAKER %s 1 %.2f %.2f <NA> <NA> spk-%s <NA>\n' %
                                 (meeting, start, duration, segment[2]))

def read_json(file_dir, name, num_files):
    """reads the json output of clustering"""
    meetings = {}
    for i in range(1, num_files + 1):
        file_path = os.path.join(file_dir, '%s.%d.json' % (name, i))
        with open(file_path, 'rb') as _json_file:
            inputjson = json.load(_json_file)['utts']
            for submeeting_name, submeeting_content in inputjson.items():
                meeting_name = submeeting_name.rsplit('-', 1)[0]
                speaker_ids = submeeting_content["output"][0]["rec_tokenid"].split()
                assert speaker_ids[-1] == str(4)
                meetings.setdefault(meeting_name, []).append(speaker_ids[:-1])
    return meetings

def match_meetings(args):
    """process json file and print the rttm file"""
    meetings_js = read_json(args.js_dir, args.js_name, args.js_num)
    meetings_scp = load_scp(args.input_scp)
    assert len(meetings_js) == len(meetings_scp), "different number of meetings"

    meetings_out = {}
    for meeting_name in meetings_scp:
        speaker_lists = meetings_js[meeting_name]
        segments = meetings_scp[meeting_name]
        segments = deque(sorted(segments, key=lambda x: x[0]))
        encompassed_indicator = deque(filter_encompassed_segments(segments, args.min_len))
        total_length = sum(~np.array(encompassed_indicator))
        print(meeting_name)
        assert total_length == sum([len(i) for i in speaker_lists]),\
            "the length of filtered sequence %d is not the same"\
            "\as the length of the speaker sequence %d"\
            %(total_length, sum([len(i) for i in speaker_lists]))

        for idx, speaker_list in enumerate(speaker_lists):
            submeeting_name = meeting_name + '-%03d' % idx
            speaker_list = deque(speaker_list)
            while speaker_list:
                encompassed = encompassed_indicator.popleft()
                segment = segments.popleft()
                if encompassed:
                    meetings_out.setdefault(submeeting_name, []).append((segment[0],
                                                                         segment[1], "0"))
                else:
                    cur_spk = speaker_list.popleft()
                    meetings_out.setdefault(submeeting_name, []).append((segment[0],
                                                                         segment[1], cur_spk))
            while True:
                if not encompassed_indicator:
                    break
                if encompassed_indicator[0] is not True:
                    break
                encompassed = encompassed_indicator.popleft()
                segment = segments.popleft()
                meetings_out.setdefault(submeeting_name, []).append((segment[0], segment[1], "0"))


    rttm_file = os.path.join(args.output_dir, args.rttm_name + '.rttm')
    write_rttm(meetings_out, rttm_file)

def main():
    """main procedure"""
    args = setup()
    match_meetings(args)

if __name__ == '__main__':
    main()
