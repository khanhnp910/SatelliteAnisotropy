jupyter nbconvert distribution_for_subhalo.ipynb --to python

search_dir=../../result/data
for filepath in "$search_dir"/*
do
    filename=$(basename $filepath)
    echo "Running for $filename"
    python distribution_for_subhalo.py "$filename"
done
