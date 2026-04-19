chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "predict_batch") {
        fetch("http://127.0.0.1:8001/predict_batch", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ texts: request.texts })
        })
        .then(response => response.json())
        .then(data => {
            sendResponse({ success: true, results: data.results });
        })
        .catch(error => {
            console.error("API Error:", error);
            sendResponse({ success: false, error: error.message });
        });

        return true;
    }
});
