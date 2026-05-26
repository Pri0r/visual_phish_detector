/*
 * fitlayout-puppeteer -- Puppeteer-based web page renderer for FitLayout
 * (c) Radek Burget 2020-2021
 *
 * Transforms a rendered web page to its JSON description that can be later
 * parsed by fitlayout-render-puppeteer.
 */

const EVAL_TIMEOUT = 30; // timeout for page script evaluation in seconds

async function scrollDown(page, maxIterations) {
    return await page.evaluate(async () => {
        let totalHeight = 0;
        await new Promise((resolve, reject) => {
			let iteration = 0;
			const distance = window.innerHeight / 2; // div 2 is for scrolling slower and let everything load
            var timer = setInterval(() => {
                const scrollHeight = document.body.scrollHeight;
                window.scrollBy({top: distance, left: 0, behavior: 'auto'});
				totalHeight += distance;
				iteration++;

                if (totalHeight >= scrollHeight || iteration > maxIterations) {
					totalHeight = scrollHeight; // for returning the maximal height in all cases
					window.scrollTo({top: 0, left: 0, behavior: 'auto'});
                    clearInterval(timer);
                    resolve();
                }
            }, 100);
		});
		return totalHeight;
    });
}

const argv = require('yargs/yargs')(process.argv.slice(2))
    .usage('Usage: $0 [options] <url>')
	//.example('$0 -W 1200 -H 800 http://cssbox.sf.net', '')
	.strictOptions(true)
    .alias('W', 'width')
    .nargs('W', 1)
	.default('W', 1200)
	.describe('W', 'Target page width')

    .alias('H', 'height')
    .nargs('H', 1)
	.default('H', 800)
	.describe('H', 'Target page height')

	.alias('P', 'persistence')
	.nargs('P', 1)
	.default('P', 1)
	.describe('P', 'Content downloading persistence: 0 (quick), 1 (standard), 2 (wait longer), 3 (get as much as possible)')

	.alias('s', 'screenshot')
	.boolean('s')
	.default('s', false)
	.describe('s', 'Include a screenshot in the result')

	.alias('I', 'download-images')
	.boolean('I')
	.default('I', false)
	.describe('I', 'Download all contained images referenced in <img> elements')

	.alias('N', 'no-headless')
	.boolean('N')
	.default('N', false)
	.describe('N', 'Do not use headless mode; show the browser in foreground')

	.alias('C', 'no-close')
	.boolean('C')
	.default('C', false)
	.describe('C', 'Do not close the browser after the operation')

	.alias('d', 'user-dir')
	.nargs('d', 1)
	.default('d', '')
	.describe('d', 'Browser profile directory to be used (default location is used when not specified)')

    .help('h')
    .alias('h', 'help')
    .argv;

if (argv._.length !== 1) {
	process.stderr.write('<url> is required. Use -h for help.\n');
	process.exit(1);
}

const targetUrl = argv._[0];
const wwidth = argv.width;
const wheight = argv.height;

let waitOptions = {};
let scrollPages = 20;
switch (argv.P) {
	case 0:
		waitOptions = {waitUntil: 'domcontentloaded', timeout: 10000};
		scrollPages = 0;
		break;
	case 1:
		waitOptions = {waitUntil: 'load', timeout: 15000};
		scrollPages = 5;
		break;
	case 2:
		waitOptions = {waitUntil: 'networkidle2', timeout: 15000};
		break;
	default:
		waitOptions = {waitUntil: 'networkidle0', timeout: 50000};
		break;
}

const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {

	const options = {
		headless: true,
		//slowMo: 250,
		args: [`--window-size=${wwidth},${wheight}`, '--no-sandbox', '--disable-web-security'], //we assume running in docker
		ignoreDefaultArgs: ['--disable-extensions'], //allow to extensions
		defaultViewport: null
	};
	if (argv.N) {
		options.headless = false;
	}
	if (argv.d !== '') {
		options.userDataDir = argv.d;
	}

	//load extensions from the 'extensions' folder if provided
	let exts = [];
	await new Promise(done => fs.readdir('extensions', (err, files) => {
		if (!err) {
			for (let file of files) {
				exts.push('extensions/' + file);
			} 
		}
		done(); 
	}));
	if (exts && exts.length > 0) {
		options.args.push('--load-extension=' + exts.join());
	}

	//launch the browser
	let lastResponse = null;
	let lastError = null;
	const browser = await puppeteer.launch(options);
	const page = await browser.newPage();
	//store the last http response
    page.on('response', async resp => {
		lastResponse = resp;
    });
	//go to the page
	try {
		await page.goto(targetUrl, waitOptions);
		// try total height as 2 window heights to make sure that everything is displayed correctly above the fold
		let totalHeight = wheight * 2;
		// if scrolling is enabled, scroll and update the total height
		if (scrollPages > 0) {
			totalHeight = await scrollDown(page, scrollPages);
		}
		await page.setViewport({width: wwidth, height: totalHeight, deviceScaleFactor: 1});
		//await page.waitForNavigation(waitOptions);
	} catch (e) {
		console.error(e);
		lastError = e;
	}
	//page.on('console', msg => console.log('PAGE LOG:', msg.text() + '\n'));

	//always take a screenshot in order to get the whole page into the viewport
	let screenShot = await page.screenshot({
		type: "png",
		fullPage: true,
		encoding: "base64"
	});

	// a task that produces the page tree
	let pageTask = page.evaluate(() => {

		/*=client.js=*/

		fitlayoutDetectLines();
		return fitlayoutExportBoxes();
	});
	// the timer that completes after 1 minute (to interrupt evaluation)
	let timerId = 0;
	let timerTask = new Promise((resolve, reject) => {
		  timerId = setTimeout(resolve, EVAL_TIMEOUT * 1000, {error: 'Evaluation timeout'});
	});
	// wait for evaluation to complete
	let pg = await Promise.race([pageTask, timerTask]);
	clearTimeout(timerId);

	// add a screenshot if it was required
	if (argv.s && screenShot !== null) {
		pg.screenshot = screenShot;
	}

	// capture the images if required
	if (argv.I && pg.images) {
		// hide the contents of the marked elemens
		await page.addStyleTag({content: '[data-fitlayoutbg="1"] * { display: none }'});
		// take the screenshots
		for (let i = 0; i < pg.images.length; i++) {
			let img = pg.images[i];
			let selector = '*[data-fitlayoutid="' + img.id + '"]';

			try {
				if (img.bg) {
					// for background images switch off the contents
					await page.$eval(selector, e => {
						e.setAttribute('data-fitlayoutbg', '1');
					});
				}

				let elem = await page.$(selector);
				if (elem !== null) {
					img.data = await elem.screenshot({
						type: "png",
						encoding: "base64"
					});
				}

				if (img.bg) {
					//for background images switch the contents on again
					await page.$eval(selector, e => {
						e.setAttribute('data-fitlayoutbg', '0');
					});
				}
			} catch (e) {
				//console.error('Couldn\'t capture image ' + i);
				//console.error(e);
			}
		}
	}

	if (!argv.C) {
		await browser.close();
	}

	if (lastResponse) {
		pg.status = lastResponse.status();
		pg.statusText = lastResponse.statusText();
	}
	if (lastError) {
		pg.error = lastError.toString();
	}

	process.stdout.write(JSON.stringify(pg));

})();
