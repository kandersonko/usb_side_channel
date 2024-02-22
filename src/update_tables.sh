#!/bin/bash

# Loop through all .tex files in the current directory that contain 'measurement' in their names
for latex_file in *measurement*.tex; do
    # Check if the file exists (in case the globbing found no matches)
    if [ ! -f "$latex_file" ]; then
        echo "No LaTeX files with 'measurement' in their names found in the current directory."
        exit 1
    fi

    # Determine the operating system to apply the correct sed syntax
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS requires an empty string argument after -i to edit in-place without backup
        sed -i '' 's/\\begin{table}/\\begin{table*}/g' "$latex_file"
        sed -i '' 's/\\end{table}/\\end{table*}/g' "$latex_file"
    else
        # Linux and other systems use -i without an additional argument for in-place editing without backup
        sed -i 's/\\begin{table}/\\begin{table*}/g' "$latex_file"
        sed -i 's/\\end{table}/\\end{table*}/g' "$latex_file"
    fi

    echo "Updated the table to span two columns in: $latex_file"
done

echo "All applicable files have been updated."
