#!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk
"""Run Spectral Clustering"""

import os
import sys
import argparse
import json
import itertools
import numpy as np
from tqdm import tqdm
import kaldiio
import utils
from SpectralCluster.spectralcluster import SpectralClusterer

def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Do speaker clsutering based on'\
                                                    'refined version of spectral clustering',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--gauss-blur', help='gaussian blur for spectral clustering',
                           type=float, default=0.1)
    cmdparser.add_argument('--p-percentile', help='p_percentile for spectral clustering',
                           type=float, default=0.95)
    cmdparser.add_argument('--custom-dist', help='e.g. euclidean, cosine', type=str, default=None)
    cmdparser.add_argument('--json-out', dest='output_json',
                           help='json output file used for scoring', default=None)
    cmdparser.add_argument('--minMaxK', nargs=2, default=[2, 4])
    cmdparser.add_argument('injson', help='ark files containing the meetings', type=str)
    cmdargs = cmdparser.parse_args()
    # setup output directory and cache commands
    if cmdargs.output_json is not None:
        outdir = os.path.dirname(cmdargs.output_json)
        utils.check_output_dir(outdir, True)
        utils.cache_command(sys.argv, outdir)
    return cmdargs

def do_spectral_clustering(dvec_list, gauss_blur=1.0, p_percentile=0.95,
                           minclusters=2, maxclusters=4, truek=4, custom_dist=None):
    """Does spectral clustering using SpectralCluster, see import"""
    if minclusters < 1 and maxclusters < 1:
        if truek == 1:
            return [0] * dvec_list.shape[0]
        clusterer = SpectralClusterer(min_clusters=truek, max_clusters=truek,
                                      p_percentile=p_percentile,
                                      gaussian_blur_sigma=gauss_blur, custom_dist=custom_dist)
    else:
        clusterer = SpectralClusterer(min_clusters=minclusters, max_clusters=maxclusters,
                                      p_percentile=p_percentile,
                                      gaussian_blur_sigma=gauss_blur, custom_dist=custom_dist)
    return clusterer.predict(dvec_list)

def permutation_invariant_seqmatch(hypothesis, reference_list):
    """For calculating segment level error rate calculation"""
    num_perm = max(4, len(set(hypothesis)))
    permutations = itertools.permutations(np.arange(num_perm))
    correct = []
    for permutation in permutations:
        mapping = {old:new for old, new in zip(np.arange(num_perm), permutation)}
        correct.append(sum([1 for hyp, ref in zip(hypothesis, reference_list)
                            if mapping[hyp] == ref]))
    return max(correct)

def evaluate_spectralclustering(args):
    """Loops through all meetings to call spectral clustering function"""
    total_correct = 0
    total_length = 0
    with open(args.injson) as _json_file:
        json_file = json.load(_json_file)
    results_dict = {}
    for midx, meeting in tqdm(list(json_file["utts"].items())):
        meeting_input = meeting["input"]
        meeting_output = meeting["output"]
        assert len(meeting_input) == 1
        assert len(meeting_output) == 1
        meeting_input = meeting_input[0]
        meeting_output = meeting_output[0]
        cur_mat = kaldiio.load_mat(meeting_input["feat"])#(samples,features)
        reference = meeting_output["tokenid"].split()
        reference = [int(ref) for ref in reference]
        assert len(reference) == cur_mat.shape[0]
        if len(reference) == 1:
            results_dict[midx] = [0]
            continue
        try:
            hypothesis = do_spectral_clustering(cur_mat,
                                                gauss_blur=args.gauss_blur,
                                                p_percentile=args.p_percentile,
                                                minclusters=int(args.minMaxK[0]),
                                                maxclusters=int(args.minMaxK[1]),
                                                truek=len(set(reference)),
                                                custom_dist=args.custom_dist)
        except:
            print("ERROR:: %s %s" % (str(reference), str(cur_mat)))
            raise
        results_dict[midx] = hypothesis
        _correct = permutation_invariant_seqmatch(hypothesis, reference)
        total_length += len(reference)
        total_correct += _correct
    print("Total Correct: %s, Total Length: %s, Percentage Correct: %s" %
          (str(total_correct), str(total_length), str(total_correct * 100 / total_length)))
    return results_dict


def write_results_dict(results_dict, output_json):
    """Writes the results dictionary into json file"""
    output_dict = {"utts":{}}
    for meeting_name, hypothesis in results_dict.items():
        hypothesis = " ".join([str(i) for i in hypothesis]) + " 4"
        output_dict["utts"][meeting_name] = {"output":[{"rec_tokenid":hypothesis}]}
    with open(output_json, 'wb') as json_file:
        json_file.write(json.dumps(output_dict, indent=4, sort_keys=True).encode('utf_8'))
    return

def main():
    """main"""
    args = setup()
    results_dict = evaluate_spectralclustering(args)
    if args.output_json is not None:
        write_results_dict(results_dict, args.output_json)

if __name__ == '__main__':
    main()
