#!/usr/bin/env fish

# Path to the Python script (adjust if needed)
set PY_SCRIPT ./notebooks/grid_search_pretrained.py

# Path to the results file - all runs append to this
set RESULTS_FILE ./grid_search_all_results_2.txt

# Number of layers range (inclusive)
set START 0
set END 30

# Optional: number of epochs override (optional)
set EPOCHS 30

# Loop: first run without Dense, then with Dense
for use_dense in 0 1
    for i in (seq $START $END)
        if test $use_dense -eq 1
            echo "Running: unfrozen_layers=$i, use_dense=TRUE"
            python $PY_SCRIPT --num-unfrozen-layers $i --use-dense --results-file $RESULTS_FILE --epochs $EPOCHS
        else
            echo "Running: unfrozen_layers=$i, use_dense=FALSE"
            python $PY_SCRIPT --num-unfrozen-layers $i --results-file $RESULTS_FILE --epochs $EPOCHS
        end

        # optional small sleep to avoid hammering I/O / letting GPU memory settle
        sleep 10
    end
end

echo "All runs finished. Results appended to $RESULTS_FILE"
