// script.js

// Utility to send POST requests and display status
function sendCommand(endpoint, successMessage) {
    fetch(endpoint, { method: "POST" })
        .then(response => response.json())
        .then(data => {
            const statusEl = document.getElementById("status");
            statusEl.textContent = data.status || successMessage;
            statusEl.style.color = "green";
        })
        .catch(error => {
            const statusEl = document.getElementById("status");
            statusEl.textContent = "Error: " + error;
            statusEl.style.color = "red";
        });
}

// Sensor buttons
document.getElementById("start-sensor").addEventListener("click", () => {
    sendCommand("/start_sensor", "Sensor simulator started.");
});

document.getElementById("stop-sensor").addEventListener("click", () => {
    sendCommand("/stop_sensor", "Sensor simulator stopped.");
});

// Inspector buttons
document.getElementById("start-inspector").addEventListener("click", () => {
    sendCommand("/start_inspector", "Inspector simulator started.");
});

document.getElementById("stop-inspector").addEventListener("click", () => {
    sendCommand("/stop_inspector", "Inspector simulator stopped.");
});

// Retrain button
document.getElementById("retrain-model-btn").addEventListener("click", () => {
    sendCommand("/retrain_model", "Retraining started.");
});

// Reset model button
document.getElementById("reset-model-btn").addEventListener("click", () =>  {
    sendCommand("/reset_model", "Model reset to initial data.");
});

// Wipe db button
document.getElementById("wipe-db-btn").addEventListener("click", () =>  {
    sendCommand("/wipe_database", "Database wiped.");
});

// Function to fetch latest predictions and update the table
function refreshTable() {
    fetch("/latest_predictions")
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector("table tbody");
            tbody.innerHTML = ""; // clear current rows

            data.products.forEach(p => {
                const row = document.createElement("tr");

                row.innerHTML = `
                    <td>${p.id}</td>
                    <td>${p.temp.toFixed(2)}</td>
                    <td>${p.humidity.toFixed(2)}</td>
                    <td>${p.sound_volume.toFixed(2)}</td>
                    <td>${p.label_predicted}</td>
                    <td>${p.label_real}</td>
                    <td>${p.accuracy_at_timestamp}</td>
                    <td>${p.timestamp}</td>
                `;
                tbody.appendChild(row);
            });
        })
        .catch(err => console.error("Failed to fetch predictions:", err));
}

// Refresh every half a second
setInterval(refreshTable, 500);

// Initial load
refreshTable();
