#!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk

import os
import sys
import argparse
import bisect
import pyhtk
import copy
import math
import json
import numpy as np
import kaldiio
from sklearn.neighbors import NearestNeighbors

IDPOS = 2
def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Get centroids of speaker d-vectors', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--input-scps', dest = 'inscps', action='append', help='the scp files of the input data "train.scp eval.scp dev.scp xxx.mlf"', type=str)
    cmdparser.add_argument('--input-mlfs', dest = 'inmlfs', action='append', help='the mlabs files of the input data "train.mlf eval.mlf dev.mlf xxx.mlf"', type=str)
    cmdparser.add_argument('--filtEncomp', help='Delete segments encompassed by another', default=False,action='store_true')
    cmdparser.add_argument('--l2norm', default=False, action='store_true', help='apply l2 normlisation to d vectors before and after averaging')
    cmdparser.add_argument('--segLenConstraint', type=int,default=None, help = 'max segment length for dvector')
    cmdparser.add_argument('--includeOrigVecs', default=False, action='store_true')
    cmdparser.add_argument('--meetingLevelDict', default=False, action='store_true', help='Two level dictionary, dict["meeting"]["speaker"] = [dvec1,dvec2,...]')
    cmdparser.add_argument('outdir', help='Output Directory for the Data', type=str, action='store')
    cmdargs = cmdparser.parse_args()
    # ensure list of scps and mlfs has the same length
    if len(cmdargs.inscps) != len(cmdargs.inmlfs):
        pyhtk.printError("number of input scps files and input mlfs has to be the same")
    for idx,scp in enumerate(cmdargs.inscps):
        cmdargs.inscps[idx] = pyhtk.getAbsPath(scp)
        if not scp.endswith('.scp'):
            pyhtk.printError('scp path has to end with .scp')
    for idx,mlf in enumerate(cmdargs.inmlfs):
        cmdargs.inmlfs[idx] = pyhtk.getAbsPath(mlf)
        if not mlf.endswith('.mlf'):
            pyhtk.printError('mlf path has to end with .mlf')
    if cmdargs.segLenConstraint is None and cmdargs.includeOrigVecs:
        pyhtk.printError("includeOrigVecs only with segLenConstraints") #code will also work otherwise, but not logical to use both
    # setup output directory and cache commands
    pyhtk.checkOutputDir(cmdargs.outdir, True)
    pyhtk.cacheCommand(sys.argv, cmdargs.outdir)
    pyhtk.changeDir(cmdargs.outdir)
    return cmdargs

def filterEncompassedSegments(_segList):
    _segList.sort(key=lambda tup: tup[1][0])
    segList = []
    #relevantSegments = []
    for idx,segment in enumerate(_segList):
        startTime = segment[1][0]
        endTime = segment[1][1]
        startBefore = [_seg for _seg in _segList if _seg[1][0] <= startTime]
        endAfter    = [_seg for _seg in _segList if _seg[1][1] >= endTime]
        startBefore.remove(segment)
        endAfter.remove(segment)
        if set(startBefore).isdisjoint(endAfter):
            segList.append(segment)
        #else:
        #    if idx < 30:
        #        print("removing %d %d" % (startTime,endTime))
         #relevantSegments.append(segment)
    return segList

def getSpeakersFromMeetings(meetings):
    speakers = set()
    for segList in meetings.values():
        for segment in segList:
            speakers.add(segment[2])
    return speakers

def applyL2Norm(cur_mat):
    return cur_mat / np.linalg.norm(cur_mat, axis=1, keepdims=True)

def getDVectorDictFromMeetings(DVectors,args,meetings):
    for meetingName,segList in meetings.items():
        print(meetingName)
        if args.filtEncomp == True:
            segList = filterEncompassedSegments(segList)
        for segment in segList:
            curspeaker = segment[2]
            cur_mat = kaldiio.load_mat(segment[0])
            cur_mat = applyL2Norm(cur_mat)
            if curspeaker not in DVectors[meetingName]:
                DVectors[meetingName][curspeaker] = []
            #if cur_mat.shape[0] >= 10:
            #    cur_mat = cur_mat[math.floor(cur_mat.shape[0]/10):-math.floor(cur_mat.shape[0]/10)]
            DVectors[meetingName][curspeaker].append(cur_mat)

def splitSegments(args,DVectors):
    splitDVectors = {}
    _numDVec = 0
    for meetingName, meeting in DVectors.items():
        splitDVectors[meetingName] = {curspeaker:[] for curspeaker in meeting}
        for curspeaker,curmats in meeting.items():
            for curmat in curmats:
                segLength = curmat.shape[0]
                numChunks = max(1,np.floor(segLength/args.segLenConstraint))
                splitMats = np.array_split(curmat,numChunks)
                assert(sum([1 for mat in splitMats if mat.shape[0] >= args.segLenConstraint*2])==0)
                splitDVectors[meetingName][curspeaker].extend(splitMats)
                if args.includeOrigVecs and numChunks != 1:
                    splitDVectors[meetingName][curspeaker].append(curmat)
                    _numDVec += 1
                _numDVec += (numChunks)
    return splitDVectors, _numDVec
def twoLevelToSingleLevelDict(twoLevelDict):
    assert type(next(iter(twoLevelDict.values()))) == dict
    singleLevelDict = {}
    for key1, value1 in twoLevelDict.items():
        for key2, value2 in value1.items():
            if key2 not in singleLevelDict.keys():
                singleLevelDict[key2] = []
            if type(value2) == list:
                singleLevelDict[key2].extend(value2)
            else:
                singleLevelDict[key2].append(value2)
    return singleLevelDict

def NPconcatenateListInDict(inputDict):
    for key1 in inputDict.keys():
        if type(inputDict[key1]) == list:
            inputDict[key1] = np.concatenate(inputDict[key1], axis=0)
        elif type(inputDict[key1]) == dict:
            for key2 in inputDict[key1].keys():
                inputDict[key1][key2] = np.concatenate(inputDict[key1][key2], axis=0)
        else:
            pyhtk.printError("something went wrong")

def DVecDictToCentroidDict(DVectors):
    centroids = {}
    for key1 in DVectors.keys():
        if type(DVectors[key1]) == np.ndarray:
            centroids[key1] = np.mean(DVectors[key1],axis=0)
        elif type(DVectors[key1]) == dict:
            if key1 not in centroids.keys():
                centroids[key1] = {}
            for key2 in DVectors[key1].keys():
                centroids[key1][key2] = np.mean(DVectors[key1][key2],axis=0)
        else:
            pyhtk.printError("something went wrong")
    return centroids

def centroidDictToOutputDict(centroidDict):
    dvectorsOut = {}
    for key1 in centroidDict.keys():
        if type(centroidDict[key1]) == np.ndarray:
            dvectorsOut[key1] = np.reshape(centroidDict[key1],[1,1,centroidDict[key1].shape[0]])
        elif type(centroidDict[key1]) == dict:
            if key1 not in dvectorsOut.keys():
                dvectorsOut[key1] = {} 
            for key2 in centroidDict[key1].keys():
                dvectorsOut[key1][key2] = np.reshape(centroidDict[key1][key2],[1,1,centroidDict[key1][key2].shape[0]])
        else:
            pyhtk.printError("something went wrong")
    return dvectorsOut

def computeCentroids(args,meetings):
    inputSize = kaldiio.load_mat(list(meetings.values())[0][0][0]).shape[1]
    print("inputSize: %d" %inputSize)
    speakerSet = getSpeakersFromMeetings(meetings)
    #DVectors is two level dictionary, dict[meetingName][curspeaker] = [mat1,mat2,...]
    DVectors = {meetingName: {} for meetingName in meetings.keys()}
    getDVectorDictFromMeetings(DVectors,args,meetings)
    
    
    _numDVec = 0
    dvectorsOut = {meetingName:{} for meetingName in DVectors.keys()}
    _DVecForStats = copy.deepcopy(DVectors)
    NPconcatenateListInDict(_DVecForStats)
    centroids = DVecDictToCentroidDict(_DVecForStats)
    if args.segLenConstraint is not None:
        DVectors,_numDVec = splitSegments(args,DVectors) 
    for meetingName,meeting in DVectors.items():
        for curspeaker,curmats in meeting.items():
            dvectorsOut[meetingName][curspeaker] = np.expand_dims(np.array([np.mean(curmat,axis=0) for curmat in curmats]),0)
    if args.l2norm:
        for meetingName in dvectorsOut:
            for curspeaker,curmats in dvectorsOut[meetingName].items():
                dvectorsOut[meetingName][curspeaker] = np.expand_dims(applyL2Norm(curmats[0]), 0)
    if args.meetingLevelDict == False:
        _dvectorsOut = dvectorsOut
        dvectorsOut = {}
        for meetingName in _dvectorsOut:
            for curspeaker,curmats in _dvectorsOut[meetingName].items():
                #curmats is (1,samples,feature)
                if curspeaker in dvectorsOut:
                    dvectorsOut[curspeaker] = np.concatenate((dvectorsOut[curspeaker],curmats),axis=1)
                else:
                    dvectorsOut[curspeaker] = curmats
    return dvectorsOut

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

        centroids = computeCentroids(args,meetings)
        np.savez(basename,**centroids)
#        with open(basename,'w') as outfile:
#            np.savez(outfile,**centroids)
                        
def main():
    args = setup()
    prepareData(args)

if __name__ == '__main__':
    main()
