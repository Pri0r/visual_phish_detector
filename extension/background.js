// API endpoint
const API_URL = "http://localhost:8000/analyze";
// abort request after specified amount of time passes
const TIMEOUT_MS = 90000;
// send analysis request after URL stays the same for this time
const DEBOUNCE_MS = 1500;
const debounceTimers = {}

// change extension icon based on the current state
function setIcon(tabId, state)
{
    const icons = { 
        loading: "blue",
        safe: "green",
        phishing: "red",
        gray: "gray" 
    };
    chrome.action.setIcon({
        path: `icons/${icons[state] || "gray"}.png`,
        tabId
    }).catch(() => {});
};

// new analysis request
async function analyzeUrl(tabId, url, isManual = false)
{
    // set state as analyzing
    setIcon(tabId, "loading");
    await chrome.storage.local.set({
        [tabId]: { status: "analyzing" }
    });

    try 
    {
        // set timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

        // send analysis request
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                url: url,
                manual: isManual
            }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        // bad response, something went wrong
        if (!response.ok) {
            let errorDetail = `HTTP ${response.status}`;
            try
            {
                const errorData = await response.json();
                if (errorData?.detail)
                    errorDetail = errorData.detail;
            }
            catch (e) {}

            setIcon(tabId, "gray");
            await chrome.storage.local.set({ [tabId]: { status: "failed", detail: errorDetail } });
            return;
        }

        // wrong response format
        const data = await response.json();
        if (!data || (data.verdict !== "phishing" && data.verdict !== "legitimate"))
            throw new Error("Invalid server response format");

        // make sure the url is still the same
        const currentTab = await chrome.tabs.get(tabId).catch(() => null);
        if (!currentTab || currentTab.url !== url)
            return;
                
        // update status
        await chrome.storage.local.set({
            [tabId]: { ...data, status: "done" }
        });
        setIcon(tabId, data.verdict === "phishing" ? "phishing" : "safe");

        // show banner if classified as phishing
        if (data.verdict === "phishing")
            chrome.tabs.sendMessage(tabId, { action: "SHOW_BANNER", data }).catch(() => {});
    } 
    catch (error)
    {
        console.error("Communication or parsing error:", error);
        setIcon(tabId, "gray");
        await chrome.storage.local.set({ [tabId]: { status: "error" } });
    }
};

// handle page navigation
chrome.tabs.onUpdated.addListener(async function(tabId, changeInfo, tab)
{
    // run only when starting to load a new page or URL changes
    if (changeInfo.status === "loading" && tab.url?.startsWith("http"))
    {
        // reset timer on URL change
        if (debounceTimers[tabId])
            clearTimeout(debounceTimers[tabId]);

        // make sure banner is removed on URL change
        chrome.tabs.sendMessage(tabId, {
            action: "REMOVE_BANNER"
        }).catch(() => {});
        
        // send new request if autoAnalyze is on
        const { autoAnalyze = false } = await chrome.storage.local.get("autoAnalyze");
        if (autoAnalyze)
        {
            setIcon(tabId, "loading");
            await chrome.storage.local.set({
                [tabId]: { status: "analyzing" }
            });

            // wait for a while, in case of redirects
            debounceTimers[tabId] = setTimeout(() => {
                analyzeUrl(tabId, tab.url, false);
                delete debounceTimers[tabId];
            }, DEBOUNCE_MS);

        }
        else
        {
            chrome.storage.local.set({
                [tabId]: {status: "idle" }
            });
            setIcon(tabId, "gray");
        }
    }
});

// handle messages
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse)
{
    // manual analysis requested
    if (request.action === "MANUAL_ANALYZE")
    {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs)
        {
            if (tabs[0])
            {
                chrome.tabs.sendMessage(tabs[0].id, { action: "REMOVE_BANNER" }).catch(() => {});
                analyzeUrl(tabs[0].id, tabs[0].url, true);
            }
        });
    }

    // return current status
    if (request.action === "CHECK_TAB_STATUS" && sender.tab)
    {
        chrome.storage.local.get([sender.tab.id.toString()]).then(res => {
            sendResponse(res[sender.tab.id.toString()] || { status: "idle" });
        });
        // keep channel open for response
        return true; 
    }
});

// keep storage claen
chrome.tabs.onRemoved.addListener(function (tabId)
{
    if (debounceTimers[tabId])
    {
        clearTimeout(debounceTimers[tabId]);
        delete debounceTimers[tabId];
    }
    chrome.storage.local.remove(tabId.toString());
});