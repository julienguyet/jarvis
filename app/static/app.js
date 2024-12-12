let mediaRecorder;
let audioChunks = [];

document.addEventListener("DOMContentLoaded", () => {
    const recordBtn = document.getElementById("record-btn");
    const stopBtn = document.getElementById("stop-btn");
    const status = document.getElementById("status");
    const userText = document.getElementById("user-text");
    const assistantText = document.getElementById("assistant-text");
    const responseAudio = document.getElementById("response-audio");

    // Update status message
    function updateStatus(message) {
        status.innerText = message;
    }

    // Start recording
    recordBtn.addEventListener("click", () => {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.start();

                recordBtn.disabled = true;
                stopBtn.disabled = false;

                updateStatus("Recording... Speak into your microphone.");
            })
            .catch(err => {
                console.error("Error accessing microphone:", err);
                updateStatus("Error accessing microphone. Please check your permissions.");
            });
    });

    // Stop recording
    stopBtn.addEventListener("click", () => {
        if (!mediaRecorder) {
            console.error("No active recorder found!");
            return;
        }

        mediaRecorder.stop();
        updateStatus("Processing your audio...");

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const formData = new FormData();
            formData.append("audio", audioBlob);

            fetch("/process_audio", { method: "POST", body: formData })
                .then(response => response.blob())
                .then(blob => {
                    const audioURL = URL.createObjectURL(blob);
                    responseAudio.src = audioURL;

                    // Update status
                    updateStatus("Response ready. Play the audio below.");
                    recordBtn.disabled = false;

                    // Placeholder for transcript update
                    userText.innerText = "Your input text here.";
                    assistantText.innerText = "AI assistant's response here.";
                })
                .catch(err => {
                    console.error("Error processing audio:", err);
                    updateStatus("An error occurred. Please try again.");
                });

            stopBtn.disabled = true;
        };
    });
});
