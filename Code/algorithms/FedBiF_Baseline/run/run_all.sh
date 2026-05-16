
SCRIPTS=(
    "fedbif.sh"
    "fedbif1.sh"
    "fedbif2.sh"
    "fedbif3.sh"
    "fedbif_femnist.sh"
    "fedbif_shakespeare.sh"
)

echo "Starting automated batch experiment pipeline"



for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "---------------------------------------------------"
    echo "Running: $script"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "---------------------------------------------------"


    bash "$script"

    status=$?
    
    if [ $status -eq 0 ]; then
        echo "$script completed successfully."
    else
        echo "$script finished with a non-zero status code ($status)."
        echo "Note: gRPC disconnection errors at the end of FL scripts can be expected; the pipeline will continue."
    fi
    
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"

    sleep 10
done

echo ""
echo "All 6 experiment scripts have finished."
