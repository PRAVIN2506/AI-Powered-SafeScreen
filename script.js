const captureButton = document.getElementById('captureButton');
const result = document.getElementById('result');
const statusMessage = document.createElement('p');
statusMessage.id = 'statusMessage';
document.querySelector('.live-demo').appendChild(statusMessage);
let intervalId;

captureButton.addEventListener('click', () => {
    if (captureButton.textContent === 'Start Demo') {
        captureButton.textContent = 'Stop Demo';
        statusMessage.textContent = 'Starting demo...';
        intervalId = setInterval(captureAndAnalyze, 3000); // Capture every 3 seconds
    } else {
        captureButton.textContent = 'Start Demo';
        statusMessage.textContent = 'Demo stopped.';
        clearInterval(intervalId);
    }
});

function captureAndAnalyze() {
    statusMessage.textContent = 'Capturing screen...';
    html2canvas(document.body).then(canvas => {
        const dataURL = canvas.toDataURL('image/png');
        statusMessage.textContent = 'Analyzing content...';
        analyzeContent(dataURL);
    });
}

function analyzeContent(dataURL) {
    // Send the image data to the server for analysis using an AJAX request
    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        result.textContent = `Prediction: ${data.prediction}, Confidence: ${data.confidence.toFixed(2)}`;
        if (data.prediction === 1 && data.confidence > 0.9) { // Assuming 1 indicates adult content
            statusMessage.textContent = 'Adult content found, redirecting...';
            window.location.href = 'https://www.coolmathgames.com/';
        } else {
            statusMessage.textContent = 'No adult content detected. Continuing...';
        }
    })
    .catch(error => {
        console.error('Error analyzing content: ', error);
        statusMessage.textContent = 'Error analyzing content.';
    });
}
