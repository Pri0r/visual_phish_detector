#!/bin/sh

# get the output folder
if [ -n "$TASK_OUT_DIR" ]; then
    OUTPUT_FOLDER="$TASK_OUT_DIR"
else
    OUTPUT_FOLDER="/app"
fi

# make sure it exists
mkdir -p "$OUTPUT_FOLDER"

# move so that the output is saved in the right place
cd "$OUTPUT_FOLDER"

# run fitlayout
java -Dfitlayout.puppeteer.backend=/app/fitlayout-puppeteer -jar /app/fitlayout/FitLayout.jar "$@"