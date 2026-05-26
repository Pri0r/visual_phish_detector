# FitLayout - Puppeteer

(c) 2015-2021 Radek Burget (burgetr@fit.vutbr.cz)


Puppeteer-based web page renderer backend for FitLayout. It uses [puppeteer](https://pptr.dev/) for driving a built-in Chromium web browser and getting the box representation of the rendered page. This package provides a backend for the FItLayout [Puppeteer renderer](https://github.com/FitLayout/FitLayout/tree/main/fitlayout-render-puppeteer).

## Requirements

`Node.js` version 14 or greater and `npm` version 6 or later are required for installing the package.

## Installation

Run the following commands for creating the `fitlayout-puppeteer` folder and configuring the backend. 

```bash
git clone https://github.com/FitLayout/fitlayout-puppeteer.git
cd fitlayout-puppeteer
npm install
```

You may run `node index.js -h` for checking that the backend works correctly. However, in normal circumstances, the package is invoked internally by the FitLayout renderer so nothing needs to be invoked manually here.

For using the puppeteer-backend, FitLayout must be provided by the path to the created `fitlayout-puppeteer` folder using the `fitlayout.puppeteer.backend` Java system property. This may be done via the Java command line (the `-D` option) or in a `config.properties` file located in the working directory as described in [Configuration](https://github.com/FitLayout/FitLayout/wiki/Installation#configuration).

The FitLayout [docker images](https://github.com/FitLayout/docker-images) contain a configured ready-to use puppeteer renderer. No further configuration is necessary when using the docker images.
