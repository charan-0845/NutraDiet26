const form = document.getElementById("predict-form");
const fileInput = document.getElementById("file-input");
const previewImage = document.getElementById("preview-image");
const previewEmpty = document.getElementById("preview-empty");
const resultsContainer = document.getElementById("results");
const statusMessage = document.getElementById("status-message");
const submitButton = document.getElementById("submit-button");
const textInput = document.querySelector("textarea");

fileInput.addEventListener("change", () => {
    const [file] = fileInput.files;
    if (!file) {
        previewImage.hidden = true;
        previewImage.src = "";
        previewEmpty.hidden = false;
        previewEmpty.textContent = "Image preview appears here";
        return;
    }

    previewImage.src = URL.createObjectURL(file);
    previewImage.hidden = false;
    previewEmpty.hidden = true;
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const [file] = fileInput.files;
    if (!file) {
        statusMessage.textContent = "Please choose an image first.";
        statusMessage.classList.add("error");
        return;
    }

    const formData = new FormData(form);
    const mealText = textInput ? textInput.value.trim() : "";
    formData.set("text", mealText);

    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";
    statusMessage.textContent = mealText
        ? "Analyzing your image and meal description..."
        : "Running the model on your image...";
    statusMessage.classList.remove("error");
    resultsContainer.innerHTML = "";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json();

        if (!response.ok || payload.error) {
            throw new Error(payload.error || "Prediction failed");
        }

        renderResults(payload.results || []);
    } catch (error) {
        statusMessage.textContent = error.message;
        statusMessage.classList.add("error");
        resultsContainer.innerHTML = '<div class="empty-state">Try another image or check that the backend is still running.</div>';
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Analyze Meal";
    }
});

function renderResults(results) {
    if (!results.length) {
        statusMessage.textContent = "No food items were detected.";
        resultsContainer.innerHTML = '<div class="empty-state">Upload a clearer image, switch modes, or include more detail in the meal description.</div>';
        return;
    }

    statusMessage.textContent = `Found ${results.length} item${results.length === 1 ? "" : "s"}.`;
    resultsContainer.innerHTML = results.map(createResultCard).join("");
}

function createResultCard(item) {
    const nutrients = item.nutrients || {};
    return `
        <article class="result-card">
            <div class="result-head">
                <h3 class="result-title">${escapeHtml(item.food || "Unknown food")}</h3>
                <span class="confidence">${formatPercent(item.confidence)}</span>
            </div>

            <div class="meta">
                <div class="metric">
                    <span class="metric-label">Detected label</span>
                    <span class="metric-value">${escapeHtml(item.food || "Unknown food")}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Matched database food</span>
                    <span class="metric-value">${escapeHtml(nutrients.matched_name || "Not found")}</span>
                </div>
            </div>

            <div class="nutrients">
                <div class="metric">
                    <span class="metric-label">Estimated weight</span>
                    <span class="metric-value">${formatNumber(item.weight)} g</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Nutrition source</span>
                    <span class="metric-value">${escapeHtml(nutrients.source || "--")}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Calories</span>
                    <span class="metric-value">${formatNumber(nutrients.calories)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Protein</span>
                    <span class="metric-value">${formatNumber(nutrients.protein)} g</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Carbs</span>
                    <span class="metric-value">${formatNumber(nutrients.carbs)} g</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Fat</span>
                    <span class="metric-value">${formatNumber(nutrients.fat)} g</span>
                </div>
            </div>
        </article>
    `;
}

function formatPercent(value) {
    if (typeof value !== "number") {
        return "0% confidence";
    }
    return `${(value * 100).toFixed(1)}% confidence`;
}

function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "--";
    }
    return Number(value).toFixed(1);
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
