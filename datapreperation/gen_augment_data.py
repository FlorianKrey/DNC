#!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk
"""Data Generation and Augmentation for Discriminative Neural Clustering"""
import os
import sys
import argparse
import json
import multiprocessing as mp
from collections import deque

import numpy as np
import kaldiio
import pyhtk


IDPOS = 2
MAXLOOPITERATIONS = 150
EPS = 10e-15
def setup():
    """Get cmds and setup directories."""
    parser = argparse.ArgumentParser(description='Prepare Data for Neural Speaker Clustering',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-scps', dest='inscps', action='append', type=str,
                        help='scp files of input data "train.scp eval.scp dev.scp xxx.mlf"')
    parser.add_argument('--input-mlfs', dest='inmlfs', action='append', type=str,
                        help='MLF files of the input data "train.mlf eval.mlf dev.mlf xxx.mlf"')
    parser.add_argument('--filtEncomp', default=False, action='store_true',
                        help='Delete segments encompassed by another')
    parser.add_argument('--maxlen', type=int, default=None,
                        help='maximum input length')
    parser.add_argument('--variableL', nargs=2, type=float, default=None,
                        help='min and max percentages of sequence length'\
                             'between which uniform sampling is used')
    parser.add_argument('--augment', type=int, default=0,
                        help='how many times to augment (0 means no augment, used for evaluation)')
    parser.add_argument('--evensplit', default=False, action='store_true',
                        help='split of meetings will be into equal chunks'\
                        'cannot be used together with variableL'\
                        'augment has to be 0, maxlen has to be set')
    parser.add_argument('--dvectordict', type=str, default=None, action=pyhtk.Abspath,
                        help='dictionary of set of dvectors to be used')
    parser.add_argument('--randomspeaker', default=False, action='store_true',
                        help='for each meeting randomise which speakers to use'\
                        'requires dvectordict')
    parser.add_argument('--maxprocesses', type=int, default=1,
                        help='number of processes in parallel')
    parser.add_argument('outdir', type=str, action='store',
                        help='Output Directory for the Data')
    cmdargs = parser.parse_args()

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
    if cmdargs.randomspeaker:
        if cmdargs.dvectordict is None:
            pyhtk.printError("if randomspeaker is used dvectordict has to be passed")
    if cmdargs.evensplit:
        if cmdargs.augment != 0:
            pyhtk.printError("When using evensplit, augment has to be 0")
        if cmdargs.maxlen is None:
            pyhtk.printError("When using evensplit, maxlen has to be set")
        if cmdargs.variableL is not None:
            pyhtk.printError("When using evensplit, variableL cannot be used")
    # setup output directory and cache commands
    pyhtk.checkOutputDir(cmdargs.outdir, True)
    pyhtk.cacheCommand(sys.argv, cmdargs.outdir)
    pyhtk.changeDir(cmdargs.outdir)
    return cmdargs

def filter_encompassed_segments(_seglist):
    """"""
    _seglist.sort(key=lambda tup: tup[1][0])
    seglist = []
    #relevantSegments = []
    for _, segment in enumerate(_seglist):
        starttime = segment[1][0]
        endtime = segment[1][1]
        start_before = [_seg for _seg in _seglist if _seg[1][0] <= starttime]
        end_after = [_seg for _seg in _seglist if _seg[1][1] >= endtime]
        start_before.remove(segment)
        end_after.remove(segment)
        if set(start_before).isdisjoint(end_after):
            seglist.append(segment)
    return seglist

def set_maxlen(args, meetinglength):
    """ based on variableL, maxlen is randomly set. If variableL is None then maxlen=args.maxlen"""
    if args.variableL is not None:
        if args.maxlen is not None:
            maxlen = int(np.random.uniform(args.variableL[0], args.variableL[1]) * args.maxlen)-1
        else:
            maxlen = int(np.random.uniform(args.variableL[0], args.variableL[1]) * meetinglength)-1
    elif args.evensplit:
        assert args.maxlen is not None
        assert args.variableL is None
        maxlen = np.ceil(meetinglength / np.ceil(meetinglength / args.maxlen))
    else:
        if args.maxlen is not None:
            maxlen = args.maxlen
        else:
            maxlen = float('inf')
    return maxlen

def AugmentSingleMeeting(args, basename, meeting_name, seg_list, dvectors, _filenames, _meetings_out, _idx):
    print(meeting_name)
    # remove back-channeling if filtEncomp is set
    seg_list.sort(key=lambda tup: tup[1][0])
    if args.filtEncomp:
        seg_list = filter_encompassed_segments(seg_list)

    # load data and process
    all_spk, all_mat = [], []
    meeting_len = 0
    for segment in seg_list:
        cur_spk = segment[2]
        cur_mat = kaldiio.load_mat(segment[0])

        # l2 norm before average
        cur_mat = cur_mat / np.linalg.norm(cur_mat, axis=1, keepdims=True)
        # average
        cur_mat = np.mean(cur_mat, axis=0, keepdims=True)
        # l2 norm after average
        cur_mat = cur_mat / np.linalg.norm(cur_mat, axis=1, keepdims=True)
        all_spk.append(cur_spk)
        all_mat.append(cur_mat)
        meeting_len += 1
    
    meetings_ark, meetings_out = {}, {}
    if args.augment >= 1:
        assert (args.maxlen is not None or args.variableL is not None), "Set maxlen or variableL"
        for i in range(args.augment):
            maxLen = set_maxlen(args,meeting_len)
            start_idx, end_idx = get_startidx(meeting_len, maxLen)
            cur_meeting_name = meeting_name + '-%03d' % i
            cur_meeting_mat = np.concatenate(all_mat[start_idx:end_idx], axis=0)
            cur_spk = all_spk[start_idx:end_idx]
            # replace cur_mat with randomly sampled d-vectors
            if dvectors is not None:
                dvec_dict = dvectors
                if args.randomspeaker:
                    spk_in_meeting = set(cur_spk)
                    # 2-level dictionary, random pick a meeting first
                    if meeting_name in dvectors:
                        all_meeting_names = list(dvectors.keys())
                        # random sample a meeting that has at least the same number of speakers as the current segment
                        sample_count = 0
                        while True:
                            sample_count += 1
                            assert sample_count < MAXLOOPITERATIONS, "possibly an infinite loop"
                            random_meeting_name = np.random.choice(all_meeting_names)
                            if len(dvectors[random_meeting_name].item()) >= len(spk_in_meeting):
                                break
                        dvec_dict = dvectors[random_meeting_name].item()
                    else:
                        assert list(spk_in_meeting)[0] in dvectors, "only 1-level or 2-lvel dictionary is allowed"
                    all_speakers = list(set(dvec_dict.keys()))
                    new_spk = np.random.choice(all_speakers, len(spk_in_meeting), replace=False)
                    spk_mapping = {orig_spk: rand_spk for orig_spk, rand_spk in zip(spk_in_meeting, new_spk)}
                    cur_spk = [spk_mapping[orig_spk] for orig_spk in cur_spk]
                samples = [np.random.choice(np.arange(dvec_dict[spk].shape[1])) for spk in cur_spk]
                cur_meeting_mat = [dvec_dict[spk][0][sample] for spk, sample in zip(cur_spk, samples)]
                cur_meeting_mat = np.array(cur_meeting_mat)
            else:
                assert args.randomspeaker is False, "cannot do random speaker with dvector dictionary"
            cur_label = get_label_from_spk(cur_spk)
            meetings_ark[cur_meeting_name] = cur_meeting_mat
            meetings_out[cur_meeting_name] = cur_meeting_mat.shape, cur_label
    else:
        #maxlen = float('inf') if args.maxlen is None else args.maxlen
        assert args.augment == 0, "invalid augment value"
        # convert lists to queues
        all_mat = deque(all_mat)
        all_spk = deque(all_spk)
        # poping out matrices until meet maxlen, form sub meeting
        segment_idx = 0
        cur_meeting_mat = []
        cur_spk = []
        cur_len = 0
        while all_mat:
            maxlen = set_maxlen(args,meeting_len)
            if cur_len == maxlen:
                cur_meeting_name = meeting_name + '-%03d' % segment_idx
                cur_meeting_mat = np.concatenate(cur_meeting_mat, axis=0)
                cur_label = get_label_from_spk(cur_spk)
                meetings_ark[cur_meeting_name] = cur_meeting_mat
                meetings_out[cur_meeting_name] = cur_meeting_mat.shape, cur_label
                segment_idx += 1
                cur_meeting_mat = []
                cur_spk = []
                cur_len = 0
            else:
                new_mat = all_mat.popleft()
                new_spk = all_spk.popleft()
                cur_meeting_mat.append(new_mat)
                cur_spk.append(new_spk)
                cur_len += 1
        cur_meeting_name = meeting_name + '-%03d' % segment_idx
        cur_meeting_mat = np.concatenate(cur_meeting_mat, axis=0)
        cur_label = get_label_from_spk(cur_spk)
        meetings_ark[cur_meeting_name] = cur_meeting_mat
        meetings_out[cur_meeting_name] = cur_meeting_mat.shape, cur_label
 
    filename = os.path.join(basename, meeting_name)
    ark_path = pyhtk.getAbsPath(filename + '.ark')
    with kaldiio.WriteHelper('ark,scp:%s,%s.scp' % (ark_path, filename)) as writer:
        for key, mat in meetings_ark.items():
            writer(key, mat)
    _filenames[_idx] = filename
    _meetings_out[_idx] = meetings_out
    print("Done with: %s" % meeting_name)

def augmentMeetingsSegments(args, meetings, basename):
    if args.dvectordict is not None:
        dvectors = dict(np.load(args.dvectordict, allow_pickle=True))
    else:
        dvectors = None

    processes = []
    manager = mp.Manager()
    _filenames = manager.list([None] * len(meetings))
    _meetings_out = manager.list([None] * len(meetings))

    for _idx, (meeting_name, seg_list) in enumerate(meetings.items()):
        fork = mp.Process(target=AugmentSingleMeeting,
                          args=(args, basename, meeting_name, seg_list, dvectors, _filenames, _meetings_out, _idx))
        fork.start()
        processes.append(fork)
        if len(processes) == args.maxprocesses or _idx == len(meetings) -1:
            for p in processes:
                p.join()
            processes = []
    print("Join finished")

    meetings_out = {}
    for meeting in _meetings_out:
        meetings_out.update(meeting)
    print("Finished Concatenating")

    # concatenate multiple scp files
    with open('%s.scp' % basename, 'w') as fout:
        for fname in _filenames:
            with open('%s.scp' %fname, 'r') as fin:
                fout.write(fin.read())
    return meetings_out

def get_startidx(meetinglength, maxlen):
    start_idx = np.random.randint(meetinglength - maxlen)
    return start_idx

def get_label_from_spk(spk_list):
    spk_mapping = {}
    for spk in spk_list:
        if spk not in spk_mapping:
            spk_mapping[spk] = len(spk_mapping)
    return [spk_mapping[spk] for spk in spk_list]

# each segment only has one label
def mlfToDict(inmlf):
    labelDict = {}
    mlfLength = len(inmlf)
    mlfIdx = 1
    while(mlfIdx < mlfLength):
        mlfLine = inmlf[mlfIdx].rstrip()
        assert('.lab"' in mlfLine)
        segName = mlfLine.split(".lab")[0].lstrip('"')
        mlfLine = inmlf[mlfIdx+1].rstrip()
        label = mlfLine.split()[2].split('[')[0]
        labelDict[segName] = label
        mlfIdx+=3
    return labelDict

def prepareData(args):
    for scp,mlf in zip(args.inscps,args.inmlfs):
        scp = os.path.abspath(scp)
        mlf = os.path.abspath(mlf)
        with open(scp) as _scp, open(mlf) as _mlf:
            inscp = list(_scp.readlines())
            inmlf = list(_mlf.readlines())
        #name of scp and ark files is based on the input scp name
        basename = os.path.splitext(os.path.basename(scp))[0]
        os.mkdir(basename)
        # get label dictionary for the segments
        labelDict = mlfToDict(inmlf)
        # key is meeting ID and value is list of tuples, ark file entries and start time of segment
        meetings = {}
        for scpline in inscp:
            segName = scpline.split()[0]
            label = labelDict[segName]
            meetingName = 'AMI-' + segName.split('-')[IDPOS]
            try:
                startTime = int(segName.split('_')[2])
                endTime = int(segName.split('_')[3])
            except:
                print(meetingName)
                raise
            if not meetingName in meetings:
                meetings[meetingName] = []
            meetings[meetingName].append((scpline.split()[1].rstrip(),(startTime,endTime),label))
        meetings = augmentMeetingsSegments(args, meetings, basename)
        # create json file
        jsonDict = {}
        jsonDict["utts"] = {}
        with open("%s.scp" % basename) as _scp:
            meetingLevelScp = {eachline.split()[0]:eachline.split()[1].rstrip() for eachline in _scp.readlines()}
        for meetingName,(shape,labelList) in meetings.items():
            inputDict = {}
            inputDict["feat"] = meetingLevelScp[meetingName]
            inputDict["name"] = "input1"
            inputDict["shape"] = shape

            outputDict = {}
            outputDict["name"] = "target1"
            outputDict["shape"] = [len(labelList),4+1]
            labelList = [str(i) for i in labelList]
            outputDict["tokenid"] = ' '.join(labelList)
            jsonDict["utts"][meetingName] = {}
            jsonDict["utts"][meetingName]["input"]  = [inputDict]
            jsonDict["utts"][meetingName]["output"] = [outputDict]
        with open("%s.json" % basename, 'wb') as json_file:
            json_file.write(json.dumps(jsonDict, indent=4, sort_keys=True).encode('utf_8'))

def main():
    args = setup()
    prepareData(args)

if __name__ == '__main__':
    main()
