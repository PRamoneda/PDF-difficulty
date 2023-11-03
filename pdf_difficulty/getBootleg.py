#!/bin/python3
import pdb
import subprocess
import os
import sys
import shutil
import numpy as np
from ExtractBootlegFeatures1 import *


def PDF2PNG(pdffile, pngfile):
    # Tries to convert PDF file to PNG files
    try:
        if not os.path.exists(os.path.dirname(pngfile)):
            os.makedirs(os.path.dirname(pngfile))
        #pdb.set_trace()
        subprocess.call(['convert','-limit','memory','32GiB','-limit','map','8GiB','-limit','disk','16GiB', '-density', '300', '-alpha', 'remove', '-resize', '2550', pdffile, f"PNG32:{pngfile}"])
        return True
    except Exception as e:
        print(e)
        return False
    return False

def hashfcn(array):
    # Encodes bootleg array to int to reduce memory
    hashNum = 0
    for i in array:
        hashNum = 2*hashNum+i
    return int(hashNum)

def PNG2Bootleg(pngDir, errorfile):
    total_bscore = []
    sortedFiles = []
    # First, sort the pages in the right order
    for subdir, dirs, files in os.walk(pngDir):
        if len(files)==1:
            sortedFiles=[files[0]]
        else:
            sortedFiles = sorted(files,key=lambda x: int(x.split('-')[-1][:-4]))
    for png in sortedFiles:
        # For each image, try to exract bootleg features
        page_bscore = np.array([]).reshape(62,0)
        imagepath = os.path.join(subdir, png)

        try:
            bscore_query = processImageFile(imagepath,errorfile)
        except:
            bscore_query=np.array([]).reshape(62,0)

        try:
            page_bscore = np.concatenate((page_bscore,bscore_query), axis=1)
        except:
            with open(error_log,'a'):
                print(input_pdf_file+"-- concatenating failed", file=f)
        # Save bootleg features in right format
        hashArray = []
        if page_bscore.shape[0]==0:
            pass
        elif page_bscore.shape[0]==1:
            hashArray = [hashfcn(page_bscore)]
        else:
            for col in page_bscore.T:
                hashArray.append(hashfcn(col))
        total_bscore.append(hashArray)

    return total_bscore

if __name__ == "__main__":
    input_pdf_file = sys.argv[1]
    tmp_file = sys.argv[2]
    output_pkl_file = sys.argv[3]
    error_log=sys.argv[4]
    # Try to convert PDF to PNG


    try:
        converted = PDF2PNG(input_pdf_file, tmp_file)
    except Exception as e:
        print(e)
        converted=False
        with open(error_log,'a') as f:
            print(input_pdf_file+"-- Failed PDF2PNG conversion for some unknown reason", file=f)
    if converted == False:
        with open(error_log,'a') as f:
            print(input_pdf_file+"-- convert command failed. Corrupted PDF file.", file=f)
    print(f"{converted} funciona 2")

    # Try to convert PNG to bootleg score
    try:
        total_bscore = PNG2Bootleg(os.path.dirname(tmp_file), error_log)
    except Exception as e:
        total_bscore = np.array([]).reshape(62, 0)
        print(e)
        with open(error_log,'a') as f:
            print(input_pdf_file+"-- Failed PNG2Bootleg conversion for some unknown reason",file=f)
            print(e)
    # Remove unnecessary files from tmp directory
    if os.path.exists(os.path.dirname(tmp_file)):
        shutil.rmtree(os.path.dirname(tmp_file))
    if not os.path.exists(os.path.dirname(output_pkl_file)):
        os.makedirs(os.path.dirname(output_pkl_file))
    with open(output_pkl_file,'wb') as f:
        pickle.dump(total_bscore,f)
