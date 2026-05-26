document.addEventListener("DOMContentLoaded", async function ()
{
    // get elements
    const element = id => document.getElementById(id);
    const statusContainer = element("status_container");
    const statusText = element("status_text");
    const scoreText = element("probability");
    const analyzeBtn = element("btn_manual");
    const autoAnalyzeCheckbox = element("auto_analyze");

    let currentTabId;

    // get auto_analyze setting (false by default)
    const { autoAnalyze = false } = await chrome.storage.local.get("autoAnalyze");
    autoAnalyzeCheckbox.checked = autoAnalyze;

    // make sure the setting is synchronized with local storage
    autoAnalyzeCheckbox.addEventListener("change", function (e)
    {
        chrome.storage.local.set({ autoAnalyze: e.target.checked });
    });

    // update UI based on the current state
    function updateUI(data)
    {
        const status = data?.status || "idle";
        const verdict = data?.verdict;

        // analyze button state
        analyzeBtn.disabled = status === "analyzing";
        analyzeBtn.textContent = status === "analyzing" ? "Čekání na server..." : "Spustit analýzu";
        analyzeBtn.style.opacity = status === "analyzing" ? "0.6" : "1";

        // reset UI
        statusContainer.className = "status_box";
        scoreText.innerHTML = "";

        // apply specific state
        if (status === "error")
        {
            statusText.textContent = "Chyba komunikace";
            scoreText.textContent = "Nelze se spojit se serverem.";
        } 
        else if (status === "failed")
        {
            statusText.textContent = "Analýza selhala";
            scoreText.textContent = data.detail ? `Důvod: ${data.detail}` : "Chyba na straně serveru.";
            scoreText.style.color = "gray";
            scoreText.style.fontSize = "0.9em";
        } 
        else if (status === "analyzing")
        {
            statusContainer.classList.add("analyzing");
            statusText.textContent = "Probíhá analýza...";
        } 
        else if (verdict === "legitimate")
        {
            statusContainer.classList.add("legit");
            statusText.textContent = "Stránka je bezpečná";
            scoreText.innerHTML = `Riziko: ${(data.risk_score * 100).toFixed(1)}%`
        } 
        else if (verdict === "phishing")
        {
            statusContainer.classList.add("phishing");
            statusText.textContent = "PODEZŘENÍ NA PHISHING!";
            let scoreHtml = `Riziko: ${(data.risk_score * 100).toFixed(1)}%`;

            // display targeted brand if known
            if (data.matched_target)
                scoreHtml += `<br>Vydává se za: <strong>${data.matched_target}</strong>`;

            // show classification reasons if known
            if (data.reasons && data.reasons.length > 0)
            {
                const reasonTransTable = {
                    "logo_identity_mismatch": "Zfalšované logo",
                    "favicon_identity_mismatch": "Zfalšovaný favicon",
                    "structural_anomaly": "Strukturální anomálie",
                    "visual_anomaly": "Vizuální anomálie",
                    "ensemble_suspicion": "Podezřelá kombinace příznaků"
                };
                const translated = data.reasons.map(r => reasonTransTable[r] || r);
                scoreHtml += `<br>Důvody: <span style="font-size: 0.9em; opacity: 0.9;">${translated.join(", ")}</span>`;
            }

            scoreText.innerHTML = scoreHtml;
        }
        else
        {
            statusContainer.classList.add("idle");
            statusText.textContent = "Neanalyzováno";
        }
    }

    // initialize UI for current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    currentTabId = tab.id;
    const result = await chrome.storage.local.get([currentTabId.toString()]);
    updateUI(result[currentTabId]);

    // listen for local storage changes
    chrome.storage.onChanged.addListener(function (changes, area)
    {
        if (area === "local" && changes[currentTabId])
            updateUI(changes[currentTabId].newValue);
    });

    // manual analysis
    analyzeBtn.addEventListener("click", function ()
    {
        updateUI({ status: "analyzing" });
        chrome.runtime.sendMessage({ action: "MANUAL_ANALYZE" });
    });
});