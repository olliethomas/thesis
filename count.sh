#!/bin/bash
dir=$1
ext=$2
totalNumberOfWords=0

for file in $(find $dir -name "*.$ext"); do
  wordCount=$(textutil -convert txt -stdout $file | wc -w)
  totalNumberOfWords=$((totalNumberOfWords + wordCount))
done  

today=$(date +%Y-%m-%d)

echo "${today}, ${totalNumberOfWords}" >> ~/Development/thesis/content/thesis_wc.csv
