#! /bin/bash
# Runs the chromium browser bundled with puppeteer
CHROME=`find node_modules/puppeteer/.local-chromium -name chrome`
echo "Running $CHROME"
$CHROME $*
