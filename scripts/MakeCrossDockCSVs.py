import argparse
import numpy as np
import pandas as pd
import time
import random
import glob
import re, os
import os.path


def topN(n, dic, thresh, redkeys=None,perpocket=False):
    '''
    This function returns the topN percentage by taking the top N poses
    and calculating the frac that meet thresh.
    Assumes dic is pocket-> key-> list of values.
    
    If redkeys (list of keys passed in) it will only use those keys
    
    If perpocket -- stats are calculated per-pocket and averaged
                    instead of just using ALL data
    '''
    
    has_stuff=[]
    counter=0 #tracks total number of poses
    
    
    if perpocket:
        for key,data in dic.items():
            kstuff=[]
            kcount=0#tracks number of poses per key
            
            for key2, data2 in data.items():
                
                #if reducing, skip over not selected items
                if redkeys and key2 not in redkeys:
                    continue
                counter+=1
                kcount+=1
                if n < len(data2):
                    lookat=n
                else:
                    lookat=len(data2)
                
                for rmsd in data2[:lookat]:
                    if rmsd < thresh:
                        kstuff.append(True)
                        break
            has_stuff.append(np.sum(kstuff)/float(kcount))
        #end for
        #print(counter)
        return np.mean(has_stuff)
    else:
        for key,data in dic.items():
            for key2, data2 in data.items():

                #if reducing, skip over not selected items
                if redkeys and key2 not in redkeys:
                    continue

                counter+=1
                if n<len(data2):
                    lookat=n
                else:
                    lookat=len(data2)
                for rmsd in data2[:lookat]:
                    if rmsd < thresh:
                        has_stuff.append(True)
                        break
        #print(counter)
        return np.sum(has_stuff)/float(counter)

def make_dict(filename,is_sweep=False,has_cnnscore=False,tag_prefix=None):
    #filling a dictionary
    # will be a dic of pocket:key:'rmsd'->[rmsds]  iff is_sweep=False
    #                             'score'->[cnnscores]
    #     else --  dic of tag:pocket:key:'rmsd'->[rmsds]
    #                                    'score'->[cnnscores]
    datadic={} #dic of pocket:key:[rmsds]
        
    with open(filename) as infile:
        for i,line in enumerate(infile):
            if i==0:
                continue
            items=line.rstrip().split(',')
            
            if has_cnnscore:
                pocket=items[6]
                key=items[7]+':'+items[8]
            else:
                pocket=items[3]
                key=items[4]+':'+items[5]
            
            rmsd=float(items[2])
            
            if is_sweep:
                if tag_prefix:
                    check=items[0].split(tag_prefix)[-1]
                else:
                    check=items[0]
                
                if '_' in check:
                    checkval=check.split('_')[0]
                    if checkval=='' or checkval=='rescore':
                        tag='0'
                    else:
                        tag=checkval
                else:
                    tag=float(items[0])
                if tag not in datadic:
                    datadic[tag]=dict()
                
                if pocket in datadic[tag] and key in datadic[tag][pocket]:
                    datadic[tag][pocket][key].append(rmsd)
                elif pocket in datadic[tag] and key not in datadic[tag][pocket]:
                    datadic[tag][pocket][key]=[rmsd]
                else:
                    datadic[tag][pocket]={key:[rmsd]}
            else:
                #no need to stratify by tag
                if pocket in datadic and key in datadic[pocket]:
                    datadic[pocket][key].append(rmsd)
                elif pocket in datadic and key not in datadic[pocket]:
                    datadic[pocket][key]=[rmsd]
                else:
                    datadic[pocket]={key:[rmsd]}
    return datadic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make CSVs for crossdock')
    #parser.add_argument('-i', '--input', nargs='+', help='Input file(s)', required=True)
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output', help='Output file', required=True)
    args = parser.parse_args()

    final_dataframe = pd.DataFrame(index=list(range(1,10)))
    for file in os.listdir(args.input):
        print(file)
        if "crossdocking_" in file:
            has_cnnscore=True
            tmp=make_dict(os.path.join(args.input,file),has_cnnscore=has_cnnscore, is_sweep=False)
            xs=[]
            for top in range(1,10):
                    val=topN(top,tmp,2,perpocket=True)
                    xs.append(val*100)
            final_dataframe[os.path.basename(file).split('.csv')[0]] = xs
    final_dataframe.to_csv(args.output)
