#!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk
"""Score rttm file against reference"""

import os
import argparse
import subprocess
import utils

def main():
    """main procedure"""
    parser = argparse.ArgumentParser(description="split rttm files into multiple submeetings")
    parser.add_argument('--score-rttm', required=True, type=str,
                        help="path to an rttm file that should be scored")
    parser.add_argument('--ref-rttm', required=True, type=str,
                        help="path to an reference rttm file")
    parser.add_argument('--output-scoredir', required=True, type=str,
                        help="path to an output scoring directory")
    args = parser.parse_args()
    utils.change_dir(os.path.dirname(os.path.realpath(__file__)))
    subprocess.call("./score_diar.sh -r %s -s %s -o %s -m notlinked"
                    % (args.ref_rttm, args.score_rttm, args.output_scoredir), shell=True)

if __name__ == '__main__':
    main()
