function showPhishingBanner(data) 
{
    // banner is already shown, do nothing
    if (document.getElementById("phishing_banner")) 
        return;

    // create dialog for the banner
    const banner = document.createElement("dialog");
    banner.id = "phishing_banner";
    
    // reset banner styles
    banner.style.cssText = "padding: 0; border: none; background: transparent; width: 100%; max-width: 100%; margin: 0; top: 0;";

    let targetHtml = "";
    if (data.matched_target)
        targetHtml = `<p style="margin-top: 5px; font-weight: bold;">Stránka se pravděpodobně vydává za službu: ${data.matched_target}</p>`;

    banner.innerHTML = `
        <div style="background-color: #ff4444; color: white; padding: 20px; text-align: center; font-family: Arial, sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h1 style="margin: 0; font-size: 24px;">Potenciálně nebezpečná stránka!</h1>
            <p>Tato stránka byla identifikována jako phishing s jistotou ${(data.risk_score * 100).toFixed(1)} %.</p>
            ${targetHtml}

            <div style="margin-top: 15px; display: inline-flex; gap: 10px;">
                <button id="btn_leave" class="btn_leave">Opustit stránku</button>
                <button id="btn_ignore" class="btn_ignore">Ignorovat riziko</button>
            </div>
        </div>

        <style>
            dialog#phishing_banner::backdrop 
            {
                background: rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(2px);
            }
            .btn_leave,
            .btn_ignore 
            {
                padding: 10px 20px;
                font-weight: bold;
                cursor: pointer;
                transition: 0.25s ease;
                background: transparent;
                border-radius: 4px;
            }
            .btn_ignore { color: white; border: 2px solid white; }
            .btn_ignore:hover { background: rgba(255,255,255,0.2); transform: scale(1.05); }
            .btn_leave { color: black; border: none; background: rgba(255,255,255,0.8); }
            .btn_leave:hover { background: rgba(255,255,255,1); transform: scale(1.05); }
        </style>
    `;
    
    document.documentElement.appendChild(banner);
    
    // prevent cancelling the dialog with escape
    banner.addEventListener('cancel', (event) => {
        event.preventDefault();
    });

    // make sure the banner is shown in the top layer so it cant be covered
    banner.showModal();

    document.getElementById("btn_leave").onclick = () => window.location.href = "about:blank";
    document.getElementById("btn_ignore").onclick = () => 
    {
        banner.close();
        banner.remove();
    };
}

function removePhishingBanner() {
    const banner = document.getElementById("phishing_banner");
    if (banner) 
    {
        banner.close();
        banner.remove();
    }
}

// handle messages
chrome.runtime.onMessage.addListener(function (request)
{
    if (request.action === "SHOW_BANNER") 
        showPhishingBanner(request.data);
    else if (request.action === "REMOVE_BANNER") 
        removePhishingBanner();
});

// check status on page load
chrome.runtime.sendMessage({action: "CHECK_TAB_STATUS" }, function (response)
{
    if (response?.verdict === "phishing") 
        showPhishingBanner(response);
});