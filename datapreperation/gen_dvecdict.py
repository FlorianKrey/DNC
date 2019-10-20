#!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk

""" Computing D-Vector Dictionary to be used for input randomisation"""
import os
import sys
import argparse
import numpy as np
import kaldiio
import pyhtk


IDPOS = 2
def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Get centroids of speaker d-vectors',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--input-scps', dest='inscps', action='append',
                           help='the scp files of the input data'\
                                '"train.scp eval.scp dev.scp xxx.mlf"', type=str)
    cmdparser.add_argument('--input-mlfs', dest='inmlfs', action='append',
                           help='the mlabs files of the input data'\
                                '"train.mlf eval.mlf dev.mlf xxx.mlf"', type=str)
    cmdparser.add_argument('--filtEncomp', help='Delete segments encompassed by another',
                           default=False, action='store_true')
    cmdparser.add_argument('--segLenConstraint', type=int, default=None,
                           help='max segment length for dvector')
    cmdparser.add_argument('--includeOrigVecs', default=False, action='store_true')
    cmdparser.add_argument('--meetingLevelDict', default=False, action='store_true',
                           help='Use two level meeting-spk dictionary'\
                                'dict["meeting"]["speaker"] = [dvec1,dvec2,...]')
    cmdparser.add_argument('outdir', help='Output Directory for the Data', type=str, action='store')
    cmdargs = cmdparser.parse_args()

    # ensure list of scps and mlfs has the same length
    if len(cmdargs.inscps) != len(cmdargs.inmlfs):
        pyhtk.printError("number of input scps files and input mlfs has to be the same")
    for idx, scp in enumerate(cmdargs.inscps):
        cmdargs.inscps[idx] = pyhtk.getAbsPath(scp)
        if not scp.endswith('.scp'):
            pyhtk.printError('scp path has to end with .scp')
    for idx, mlf in enumerate(cmdargs.inmlfs):
        cmdargs.inmlfs[idx] = pyhtk.getAbsPath(mlf)
        if not mlf.endswith('.mlf'):
            pyhtk.printError('mlf path has to end with .mlf')
    if cmdargs.segLenConstraint is None and cmdargs.includeOrigVecs:
        #code will also work otherwise, but not logical
        pyhtk.printError("includeOrigVecs only with segLenConstraints")
    # setup output directory and cache commands
    pyhtk.checkOutputDir(cmdargs.outdir, True)
    pyhtk.cacheCommand(sys.argv, cmdargs.outdir)
    pyhtk.changeDir(cmdargs.outdir)
    return cmdargs

def filter_encompassed_segments(_seg_list):
    """remove segments completely contained within another one"""
    _seg_list.sort(key=lambda tup: tup[1][0])
    seg_list = []
    for _, segment in enumerate(_seg_list):
        start_time = segment[1][0]
        end_time = segment[1][1]
        start_before = [_seg for _seg in _seg_list if _seg[1][0] <= start_time]
        end_after = [_seg for _seg in _seg_list if _seg[1][1] >= end_time]
        start_before.remove(segment)
        end_after.remove(segment)
        if set(start_before).isdisjoint(end_after):
            seg_list.append(segment)
    return seg_list

def l2_normalise_matrix(cur_mat):
    """apply l2 normalisation to vectors of matrix"""
    return cur_mat / np.linalg.norm(cur_mat, axis=1, keepdims=True)

def get_dvector_dict_from_meetings(dvectors, args, meetings):
    """
        create two-level dictonary dict[meeting_name][speaker_id]
        value is list of numpy matrices
        the matrices are the dvectors of the segments
    """
    for meeting_name, seg_list in meetings.items():
        if args.filtEncomp is True:
            seg_list = filter_encompassed_segments(seg_list)
        for segment in seg_list:
            curspeaker = segment[2]
            cur_mat = kaldiio.load_mat(segment[0])
            if curspeaker not in dvectors[meeting_name]:
                dvectors[meeting_name][curspeaker] = []
            dvectors[meeting_name][curspeaker].append(cur_mat)

def split_segments(args, dvectors):
    """ Splits long segments into multiple shorter once of equal length """
    split_dvectors = {}
    for meeting_name, meeting in dvectors.items():
        split_dvectors[meeting_name] = {curspeaker:[] for curspeaker in meeting}
        for curspeaker, curmats in meeting.items():
            for curmat in curmats:
                seg_length = curmat.shape[0]
                num_chunks = max(1, np.floor(seg_length/args.segLenConstraint))
                split_mats = np.array_split(curmat, num_chunks)
                assert not [1 for mat in split_mats if mat.shape[0] >= args.segLenConstraint*2]
                split_dvectors[meeting_name][curspeaker].extend(split_mats)
                if args.includeOrigVecs and num_chunks != 1:
                    split_dvectors[meeting_name][curspeaker].append(curmat)
    return split_dvectors

def two_level_to_single_level_dict(two_level_dict):
    """
        Converts a two-level dictionary to a single level dictionary
        dict[meeting][speaker] --> dict[speaker]
    """
    assert isinstance((next(iter(two_level_dict.values()))), dict)
    single_level_dict = {}
    for _, value1 in two_level_dict.items():
        for key2, value2 in value1.items():
            if key2 not in single_level_dict.keys():
                single_level_dict[key2] = []
            if isinstance(value2, list):
                single_level_dict[key2].extend(value2)
            else:
                single_level_dict[key2].append(value2)
    return single_level_dict

def concatenate_list_in_dict(inputdict):
    """
        single or two level dictionary with lists of np arrays as values
        each list of arrays will be concatenated
    """
    for key1 in inputdict.keys():
        if isinstance(inputdict[key1], list):
            inputdict[key1] = np.concatenate(inputdict[key1], axis=0)
        elif isinstance(inputdict[key1], dict):
            for key2 in inputdict[key1].keys():
                inputdict[key1][key2] = np.concatenate(inputdict[key1][key2], axis=0)
        else:
            pyhtk.printError("something went wrong")

def generate_dvecdict(args, meetings):
    """ returns dictionary of segment level dvectors keys can be speakers or meetings """
    # dvectors is two level dictionary, dict[meeting_name][curspeaker] = [mat1,mat2,...]
    dvectors = {meeting_name: {} for meeting_name in meetings.keys()}
    get_dvector_dict_from_meetings(dvectors, args, meetings)

    dvectors_out = {meeting_name:{} for meeting_name in dvectors.keys()}
    if args.segLenConstraint is not None:
        dvectors = split_segments(args, dvectors)
    for meeting_name, meeting in dvectors.items():
        for curspeaker, curmats in meeting.items():
            # l2 normalise then average
            dvectors_out[meeting_name][curspeaker] = \
                np.array([np.mean(l2_normalise_matrix(curmat), axis=0)
                          for curmat in curmats])
    # l2 normalise again
    for meeting_name in dvectors_out:
        for curspeaker, curmats in dvectors_out[meeting_name].items():
            dvectors_out[meeting_name][curspeaker] = \
                l2_normalise_matrix(curmats)
    if args.meetingLevelDict is False:
        _dvectors_out = dvectors_out
        dvectors_out = {}
        for meeting_name in _dvectors_out:
            for curspeaker, curmats in _dvectors_out[meeting_name].items():
                #curmats is (1,samples,feature)
                if curspeaker in dvectors_out:
                    dvectors_out[curspeaker] = \
                        np.concatenate((dvectors_out[curspeaker], curmats), axis=0)
                else:
                    dvectors_out[curspeaker] = curmats
    return dvectors_out

# each segment only has one label
def mlf_to_dict(inmlf):
    """ loads htk-style mlf file into dictionary """
    label_dict = {}
    mlf_length = len(inmlf)
    mlf_idx = 1
    while mlf_idx < mlf_length:
        mlfline = inmlf[mlf_idx].rstrip()
        assert '.lab"' in mlfline
        segname = mlfline.split(".lab")[0].lstrip('"')
        mlfline = inmlf[mlf_idx+1].rstrip()
        label = mlfline.split()[2].split('[')[0]
        label_dict[segname] = label
        mlf_idx += 3
    return label_dict

def prepare_data(args):
    """first loads scp and mlf files for further processing"""
    for scp, mlf in zip(args.inscps, args.inmlfs):
        scp = os.path.abspath(scp)
        mlf = os.path.abspath(mlf)
        with open(scp) as _scp, open(mlf) as _mlf:
            inscp = list(_scp.readlines())
            inmlf = list(_mlf.readlines())
        # name of dictionary will be based on scp file name
        basename = os.path.splitext(os.path.basename(scp))[0]
        # get label dictionary for the segments
        label_dict = mlf_to_dict(inmlf)
        # key is meeting ID and value is list of tuples, ark file entries and start time of segment
        meetings = {}
        for scpline in inscp:
            segname = scpline.split()[0]
            label = label_dict[segname]
            meeting_name = 'AMI-' + segname.split('-')[IDPOS]
            try:
                start_time = int(segname.split('_')[2])
                end_time = int(segname.split('_')[3])
            except:
                print(meeting_name)
                raise
            if not meeting_name in meetings:
                meetings[meeting_name] = []
            meetings[meeting_name].append(
                (scpline.split()[1].rstrip(), (start_time, end_time), label))

        dvecdict = generate_dvecdict(args, meetings)
        np.savez(basename, **dvecdict)

def main():
    """main"""
    args = setup()
    prepare_data(args)

if __name__ == '__main__':
    main()
