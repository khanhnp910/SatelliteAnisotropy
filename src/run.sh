jupyter nbconvert main.ipynb --to python

search_dir=../../result/data
for filepath in "$search_dir"/*
do
    filename=$(basename $filepath)
    echo "Running for $filename"
    python main.py "$filename"
done
