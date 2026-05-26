# Visual Phishing Detector
Phishing detection system based on visual website similarity. Combines deep learning with brand identity verification using logos and favicons.

## Usage

1. Start the server:
```bash
docker compose up --build
```

2. Submit an analysis request:
   - using the web interface at http://localhost:8000
   - using the browser extension:
      - in your browser, open Manage Extensions
      - enable Developer Mode if required
      - choose Load unpacked
      - select the `/extension` directory

      New analysis can be started from the extension popup window.

3. Results will be displayed both in the server console and on the client side.


## Third-Party Tools
- [FitLayout](https://github.com/FitLayout/FitLayout) (`/fitlayout`)
- [fitlayout-puppeteer](https://github.com/FitLayout/fitlayout-puppeteer) (`/fitlayout-puppeteer`)