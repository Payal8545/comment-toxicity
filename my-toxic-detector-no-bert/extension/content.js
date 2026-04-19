let pendingNodes = [];
let isProcessing = false;

// Process nodes in batches
async function processBatch() {
    if (isProcessing || pendingNodes.length === 0) return;
    
    isProcessing = true;
    
    // Take up to 50 nodes at a time to not overwhelm the frontend or backend
    const batch = pendingNodes.splice(0, 50);
    const texts = batch.map(item => item.text);
    
    try {
        // Send to background to bypass CSP limitation
        const response = await chrome.runtime.sendMessage({
            action: "predict_batch",
            texts: texts
        });
        
        if (response && response.success) {
            for (let i = 0; i < response.results.length; i++) {
                const result = response.results[i];
                const nodeInfo = batch[i];
                
                if (result.is_toxic) {
                    nodeInfo.element.style.filter = "blur(5px)";
                    nodeInfo.element.style.transition = "filter 0.3s ease-in-out";
                    
                    // Add a click-to-reveal feature
                    nodeInfo.element.title = "Toxic content detected. Click to reveal.";
                    nodeInfo.element.style.cursor = "pointer";
                    
                    nodeInfo.element.addEventListener("click", function revealText(event) {
                        // Unblur the text
                        nodeInfo.element.style.filter = "none";
                        nodeInfo.element.style.cursor = "auto";
                        nodeInfo.element.title = "";
                        
                        // Prevent accidental clicks on links inside the text
                        event.preventDefault(); 
                        event.stopPropagation();
                        
                        // Remove the event listener so it behaves like normal text again
                        nodeInfo.element.removeEventListener("click", revealText);
                    });
                }
            }
        }
    } catch (err) {
        console.error("Failed to check batch for toxicity", err);
    } finally {
        isProcessing = false;
        
        // If there are still pending items, after a brief pause do the next batch
        if (pendingNodes.length > 0) {
            setTimeout(processBatch, 200);
        }
    }
}

// Function to safely check if text shouldn't be ignored
function isValidText(text) {
    if (!text) return false;
    text = text.trim();
    // Has a reasonable length and contains alphabetical characters
    return text.length > 5 && /[a-zA-Z]/.test(text);
}

let batchTimeout = null;

// Search inside element for text and queue it
function evaluateElement(element) {
    // Avoid double checking elements we already processed
    if (element.dataset.toxicityChecked === "yes") return;
    
    // CRITICAL PERFORMANCE FIX: Use textContent instead of innerText
    // innerText forces the browser to calculate the physical layout geometry of the page.
    // Doing this hundreds of times per second on YouTube causes massive stuttering/interruption.
    const text = element.textContent;
    if (isValidText(text)) {
        element.dataset.toxicityChecked = "yes";
        pendingNodes.push({ element: element, text: text.trim() });
        
        // CRITICAL PERFORMANCE FIX: Debounce the processing trigger!
        // Prevents generating 1,000 parallel timeout timers when YouTube loads 1,000 comments instantly.
        if (!batchTimeout) {
            batchTimeout = setTimeout(() => {
                batchTimeout = null;
                processBatch();
            }, 500); 
        }
    }
}

// Check early DOM node existence
function initialScan() {
    // Target common textual tags, adding YouTube's custom text tag!
    const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, blockquote, yt-formatted-string');
    elements.forEach(evaluateElement);
    
    // Attempting safely target loose text elements (leaf nodes without children basically)
    const looseSpans = document.querySelectorAll('span, div');
    looseSpans.forEach(el => {
         if (el.children.length === 0 && isValidText(el.innerText)) {
             evaluateElement(el);
         }
    });
}

// Watch the DOM for new elements (infinite scrolling comments like Reddit/Youtube/Twitter)
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        // If an app injects text into an existing node AFTER creating it (Data-binding)
        if (mutation.type === 'characterData') {
            const parent = mutation.target.parentElement;
            if (parent && ['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'BLOCKQUOTE', 'SPAN', 'YT-FORMATTED-STRING'].includes(parent.tagName)) {
                // Remove the cache lock and re-evaluate
                parent.dataset.toxicityChecked = "no";
                evaluateElement(parent);
            }
        }
        
        // If an app adds entirely new elements to the screen
        if (mutation.type === 'childList') {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // If it's a structural text container, check it
                    if (['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'BLOCKQUOTE', 'SPAN', 'YT-FORMATTED-STRING'].includes(node.tagName)) {
                        evaluateElement(node);
                    }
                    
                    // Also recursively check its children
                    const children = node.querySelectorAll('p, h1, h2, h3, h4, h5, h6, blockquote, span, yt-formatted-string');
                    children.forEach(child => {
                        if (['SPAN'].includes(child.tagName) && child.children.length > 0) return;
                        evaluateElement(child);
                    });
                }
            });
        }
    });
});

// Start the observer listening to changes globally (childList = additions, characterData = text changing)
observer.observe(document.body, { childList: true, subtree: true, characterData: true });

// Run the first scan
setTimeout(initialScan, 200);
