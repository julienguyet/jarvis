let mediaRecorder;
let audioChunks = [];

// Initialize DOM elements after the page loads
document.addEventListener("DOMContentLoaded", () => {
    const recordBtn = document.getElementById("record-btn");
    const stopBtn = document.getElementById("stop-btn");
    const status = document.getElementById("status");
    const responseAudio = document.getElementById("response-audio");
    const userMessageInput = document.getElementById("user-message");
    const sendBtn = document.getElementById("send-btn");
    const transcript = document.getElementById("transcript");

    // Helper function: Update the status message
    function updateStatus(message) {
        status.innerText = message;
    }

    // Helper function: Add message to the conversation history
    function addToTranscript(role, text) {
        const messageElement = document.createElement("p");
        messageElement.innerHTML = `<strong>${role}:</strong> ${text}`;
        transcript.appendChild(messageElement);
        transcript.scrollTop = transcript.scrollHeight; // Auto-scroll to the latest message
    }

    // Audio interaction: Start recording
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

    // Audio interaction: Stop recording
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
                .then(response => response.json())
                .then(data => {
                    const { user_text, assistant_text, audio_url } = data;

                    // Update transcript and play audio
                    addToTranscript("You", user_text);
                    addToTranscript("Assistant", assistant_text);
                    responseAudio.src = audio_url;
                    responseAudio.play();

                    // Reset UI
                    updateStatus("Response ready. Play the audio below.");
                    recordBtn.disabled = false;
                    stopBtn.disabled = true;
                })
                .catch(err => {
                    console.error("Error processing audio:", err);
                    updateStatus("An error occurred. Please try again.");
                });
        };
    });

    // Text interaction: Send message
    sendBtn.addEventListener("click", () => {
        const userMessage = userMessageInput.value.trim();
        if (!userMessage) {
            updateStatus("Please type a message before sending.");
            return;
        }

        // Add user's message to transcript and clear input field
        addToTranscript("You", userMessage);
        userMessageInput.value = "";
        updateStatus("Processing your message...");

        // Ensure the key is 'message' instead of 'text' when sending data to the server
        fetch("/process_chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage })  // changed 'text' to 'message'
        })
            .then(response => response.json())
            .then(data => {
                const { assistant_text } = data;

                // Add assistant's response to transcript
                addToTranscript("Assistant", assistant_text);
                updateStatus("Response received.");
            })
            .catch(err => {
                console.error("Error processing text:", err);
                updateStatus("An error occurred. Please try again.");
            });
    });
});
