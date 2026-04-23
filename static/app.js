let currentResults = [];
const debounceTimers = {};

const form = document.getElementById("predict-form");
const fileInput = document.getElementById("file-input");
const previewImage = document.getElementById("preview-image");
const previewEmpty = document.getElementById("preview-empty");
const resultsContainer = document.getElementById("results");
const summaryContainer = document.getElementById("overall-summary");
const statusMessage = document.getElementById("status-message");
const submitButton = document.getElementById("submit-button");
const textInput = document.querySelector("textarea");

renderEmptySummary();

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
        setStatus("Please choose an image first.", true);
        return;
    }

    const formData = new FormData(form);
    const mealText = textInput ? textInput.value.trim() : "";
    formData.set("text", mealText);

    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";
    setStatus(
        mealText ? "Analyzing image and meal description..." : "Running the model on your image...",
        false
    );
    resultsContainer.innerHTML = "";
    renderEmptySummary("Analysis in progress...");

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const payload = await response.json();

        if (!response.ok || payload.error) {
            throw new Error(payload.error || "Prediction failed");
        }

        currentResults = (payload.results || []).map((result, index) => ({
            ...result,
            _id: index,
            _nutrientsPer100g: computePer100g(result.nutrients, result.weight),
        }));

        renderAll();
    } catch (error) {
        setStatus(error.message, true);
        resultsContainer.innerHTML = '<div class="empty-state">Try another image or check that the backend is running.</div>';
        renderEmptySummary("No totals yet");
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Analyze Meal";
    }
});

function setStatus(message, isError) {
    statusMessage.textContent = message;
    statusMessage.classList.toggle("error", isError);
}

function numericOrNull(value) {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue : null;
}

function scaleValue(value, factor) {
    const numberValue = numericOrNull(value);
    return numberValue === null ? null : numberValue * factor;
}

function computePer100g(nutrients, weight) {
    const numberWeight = numericOrNull(weight);
    if (!nutrients || !numberWeight || numberWeight <= 0) {
        return nutrients || {};
    }

    const factor = 100 / numberWeight;
    return {
        ...nutrients,
        calories: scaleValue(nutrients.calories, factor),
        protein: scaleValue(nutrients.protein, factor),
        carbs: scaleValue(nutrients.carbs, factor),
        fat: scaleValue(nutrients.fat, factor),
    };
}

function scaleNutrients(base, newWeight) {
    const numberWeight = numericOrNull(newWeight);
    const factor = numberWeight ? numberWeight / 100 : 0;
    return {
        ...base,
        calories: scaleValue(base.calories, factor),
        protein: scaleValue(base.protein, factor),
        carbs: scaleValue(base.carbs, factor),
        fat: scaleValue(base.fat, factor),
    };
}

function fmt(value) {
    const numberValue = numericOrNull(value);
    return numberValue === null ? "--" : numberValue.toFixed(1);
}

function pct(value) {
    const numberValue = numericOrNull(value);
    return numberValue === null ? "0%" : `${(numberValue * 100).toFixed(1)}%`;
}

function esc(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function renderAll() {
    if (!currentResults.length) {
        setStatus("No food items. Analyze a new image to start again.", false);
        resultsContainer.innerHTML = '<div class="empty-state">No items remain.</div>';
        renderEmptySummary("No food items selected");
        return;
    }

    setStatus(
        `Found ${currentResults.length} item${currentResults.length === 1 ? "" : "s"}. Edit names or weights to update totals.`,
        false
    );
    resultsContainer.innerHTML = currentResults.map(createResultCard).join("");
    renderSummary();
    attachCardListeners();
}

function getNutrientUnit(key) {
    const unitMap = {
        'energy_kcal': 'kcal', 'protein': 'g', 'carbs': 'g', 'fat': 'g',
        'saturated_fat': 'g', 'monounsaturated_fat': 'g', 'polyunsaturated_fat': 'g',
        'fiber': 'g', 'sugars': 'g', 'cholesterol': 'mg',
        'calcium': 'mg', 'phosphorus': 'mg', 'magnesium': 'mg', 'sodium': 'mg', 'potassium': 'mg',
        'iron': 'mg', 'copper': 'mg', 'selenium': 'mcg', 'zinc': 'mg',
        'vitamin_a': 'mcg', 'vitamin_c': 'mg', 'vitamin_e': 'mg',
        'thiamin': 'mg', 'riboflavin': 'mg', 'niacin': 'mg', 'vitamin_b6': 'mg', 'folate': 'mcg'
    };
    return unitMap[key] || '';
}

function getNutrientName(key) {
    const nameMap = {
        'energy_kcal': 'Energy', 'protein': 'Protein', 'carbs': 'Carbohydrates', 'fat': 'Total Fat',
        'saturated_fat': 'Saturated Fat', 'monounsaturated_fat': 'Monounsaturated Fat', 'polyunsaturated_fat': 'Polyunsaturated Fat',
        'fiber': 'Fiber', 'sugars': 'Sugars', 'cholesterol': 'Cholesterol',
        'calcium': 'Calcium', 'phosphorus': 'Phosphorus', 'magnesium': 'Magnesium', 'sodium': 'Sodium', 'potassium': 'Potassium',
        'iron': 'Iron', 'copper': 'Copper', 'selenium': 'Selenium', 'zinc': 'Zinc',
        'vitamin_a': 'Vitamin A', 'vitamin_c': 'Vitamin C', 'vitamin_e': 'Vitamin E',
        'thiamin': 'Thiamine (B1)', 'riboflavin': 'Riboflavin (B2)', 'niacin': 'Niacin (B3)', 'vitamin_b6': 'Vitamin B6', 'folate': 'Folate'
    };
    return nameMap[key] || key;
}

function createNutrientRows(nutrients) {
    // Key nutrients to show always
    const keyNutrients = ['energy_kcal', 'protein', 'carbs', 'fat', 'fiber'];
    
    // All possible nutrients
    const allNutrients = [
        'energy_kcal', 'protein', 'carbs', 'fat', 'saturated_fat', 'monounsaturated_fat', 'polyunsaturated_fat',
        'fiber', 'sugars', 'cholesterol',
        'calcium', 'phosphorus', 'magnesium', 'sodium', 'potassium',
        'iron', 'copper', 'selenium', 'zinc',
        'vitamin_a', 'vitamin_c', 'vitamin_e', 'thiamin', 'riboflavin', 'niacin', 'vitamin_b6', 'folate'
    ];
    
    // Build key nutrients display
    let keyHtml = '';
    for (const key of keyNutrients) {
        const value = nutrients[key];
        if (value !== null && value !== undefined) {
            const unit = getNutrientUnit(key);
            const name = getNutrientName(key);
            keyHtml += `<div class="metric"><span class="metric-label">${name}</span><span class="metric-value">${fmt(value)} ${unit}</span></div>`;
        }
    }
    
    // Build additional nutrients display (for expandable section)
    let extraHtml = '';
    for (const key of allNutrients) {
        if (!keyNutrients.includes(key)) {
            const value = nutrients[key];
            if (value !== null && value !== undefined) {
                const unit = getNutrientUnit(key);
                const name = getNutrientName(key);
                extraHtml += `<div class="metric"><span class="metric-label">${name}</span><span class="metric-value">${fmt(value)} ${unit}</span></div>`;
            }
        }
    }
    
    return { keyHtml, extraHtml };
}

function createResultCard(item) {
    const nutrients = item.nutrients || {};
    const id = item._id;
    const { keyHtml, extraHtml } = createNutrientRows(nutrients);
    const hasExtra = extraHtml.length > 0;

    return `
<article class="result-card" data-id="${id}">
    <div class="result-head">
        <div class="result-title-wrap">
            <input
                class="food-name-input"
                type="text"
                value="${esc(item.food || "")}"
                data-id="${id}"
                title="Edit the food name, then press Enter or click away"
                placeholder="Food name"
            >
            <span class="confidence">${pct(item.confidence)} confidence</span>
        </div>
        <button class="remove-btn" type="button" data-id="${id}" title="Remove this item">Remove</button>
    </div>

    <div class="edit-hint">Edit the name, press Enter, or change the weight. Totals update automatically.</div>

    <div class="card-meta">
        <div class="metric">
            <span class="metric-label">Matched in database</span>
            <span class="metric-value matched-name">${esc(nutrients.matched_name || "Not found")}</span>
        </div>
        <div class="metric weight-metric">
            <span class="metric-label">Weight</span>
            <span class="metric-value weight-row">
                <input
                    class="weight-input"
                    type="number"
                    min="1"
                    max="5000"
                    step="1"
                    value="${fmt(item.weight)}"
                    data-id="${id}"
                    aria-label="Weight in grams"
                > g
            </span>
        </div>
    </div>

    <div class="nutrients" id="nutrients-${id}">
        ${keyHtml}
    </div>
    
    ${hasExtra ? `
    <div class="nutrients-toggle-wrap">
        <button class="nutrients-toggle" data-id="${id}" type="button">
            Show All Nutrients (27 Total)
        </button>
    </div>
    <div class="nutrients-extra" id="nutrients-extra-${id}" style="display:none;">
        <div class="nutrients-divider">Additional Nutrients</div>
        ${extraHtml}
    </div>
    ` : ''}
</article>`;
}

function renderEmptySummary(message = "Run an analysis to see meal totals") {
    summaryContainer.innerHTML = `
<div class="summary-card">
    <div class="summary-header">
        <p class="panel-kicker">Overall Nutrition</p>
        <h2 class="summary-title">Meal Totals</h2>
        <p class="summary-sub">${esc(message)}</p>
    </div>

    <div class="summary-hero">
        <div class="summary-cal-block">
            <span class="cal-number">--</span>
            <span class="cal-label">kcal</span>
        </div>
        <div class="macro-chart-wrap">
            <div class="macro-bar">
                <div class="macro-seg seg-protein" style="width:0%" title="Protein 0%"></div>
                <div class="macro-seg seg-carbs" style="width:0%" title="Carbs 0%"></div>
                <div class="macro-seg seg-fat" style="width:0%" title="Fat 0%"></div>
            </div>
            <div class="macro-legend">
                <span><i class="legend-dot dot-protein"></i>Protein --</span>
                <span><i class="legend-dot dot-carbs"></i>Carbs --</span>
                <span><i class="legend-dot dot-fat"></i>Fat --</span>
            </div>
        </div>
    </div>

    <div class="summary-grid">
        <div class="summary-metric">
            <span class="summary-label">Protein</span>
            <span class="summary-value">-- g</span>
        </div>
        <div class="summary-metric">
            <span class="summary-label">Carbohydrates</span>
            <span class="summary-value">-- g</span>
        </div>
        <div class="summary-metric">
            <span class="summary-label">Fat</span>
            <span class="summary-value">-- g</span>
        </div>
        <div class="summary-metric">
            <span class="summary-label">Total Weight</span>
            <span class="summary-value">-- g</span>
        </div>
    </div>
</div>`;
}

function renderSummary() {
    const totals = currentResults.reduce(
        (acc, item) => {
            const nutrients = item.nutrients || {};
            acc.calories += numericOrNull(nutrients.calories) || 0;
            acc.protein += numericOrNull(nutrients.protein) || 0;
            acc.carbs += numericOrNull(nutrients.carbs) || 0;
            acc.fat += numericOrNull(nutrients.fat) || 0;
            acc.weight += numericOrNull(item.weight) || 0;
            return acc;
        },
        { calories: 0, protein: 0, carbs: 0, fat: 0, weight: 0 }
    );

    const proteinCalories = totals.protein * 4;
    const carbCalories = totals.carbs * 4;
    const fatCalories = totals.fat * 9;
    const macroCalories = proteinCalories + carbCalories + fatCalories || 1;

    const proteinPct = Math.round((proteinCalories / macroCalories) * 100);
    const carbsPct = Math.round((carbCalories / macroCalories) * 100);
    const fatPct = Math.max(0, 100 - proteinPct - carbsPct);

    summaryContainer.innerHTML = `
<div class="summary-card">
    <div class="summary-header">
        <p class="panel-kicker">Overall Nutrition</p>
        <h2 class="summary-title">Meal Totals</h2>
        <p class="summary-sub">${currentResults.length} item${currentResults.length === 1 ? "" : "s"} &middot; ${fmt(totals.weight)} g total</p>
    </div>

    <div class="summary-hero">
        <div class="summary-cal-block">
            <span class="cal-number">${fmt(totals.calories)}</span>
            <span class="cal-label">kcal</span>
        </div>
        <div class="macro-chart-wrap">
            <div class="macro-bar">
                <div class="macro-seg seg-protein" style="width:${proteinPct}%" title="Protein ${proteinPct}%"></div>
                <div class="macro-seg seg-carbs" style="width:${carbsPct}%" title="Carbs ${carbsPct}%"></div>
                <div class="macro-seg seg-fat" style="width:${fatPct}%" title="Fat ${fatPct}%"></div>
            </div>
            <div class="macro-legend">
                <span><i class="legend-dot dot-protein"></i>Protein ${proteinPct}%</span>
                <span><i class="legend-dot dot-carbs"></i>Carbs ${carbsPct}%</span>
                <span><i class="legend-dot dot-fat"></i>Fat ${fatPct}%</span>
            </div>
        </div>
    </div>

    <div class="summary-grid">
        <div class="summary-metric">
            <span class="summary-label">Protein</span>
            <span class="summary-value">${fmt(totals.protein)} g</span>
        </div>
        <div class="summary-metric">
            <span class="summary-label">Carbohydrates</span>
            <span class="summary-value">${fmt(totals.carbs)} g</span>
        </div>
        <div class="summary-metric">
            <span class="summary-label">Fat</span>
            <span class="summary-value">${fmt(totals.fat)} g</span>
        </div>
        <div class="summary-metric">
            <span class="summary-label">Total Weight</span>
            <span class="summary-value">${fmt(totals.weight)} g</span>
        </div>
    </div>
</div>`;
}

function attachCardListeners() {
    document.querySelectorAll(".remove-btn").forEach((button) => {
        button.addEventListener("click", () => {
            const id = Number(button.dataset.id);
            currentResults = currentResults.filter((result) => result._id !== id);
            renderAll();
        });
    });

    document.querySelectorAll(".weight-input").forEach((input) => {
        input.addEventListener("input", () => {
            const id = Number(input.dataset.id);
            clearTimeout(debounceTimers[id]);
            debounceTimers[id] = setTimeout(() => {
                const newWeight = Number(input.value);
                if (!Number.isFinite(newWeight) || newWeight <= 0) {
                    return;
                }
                applyWeightChange(id, newWeight);
            }, 300);
        });
    });

    document.querySelectorAll(".food-name-input").forEach((input) => {
        const trigger = async () => {
            const id = Number(input.dataset.id);
            const newName = input.value.trim();
            if (!newName) {
                return;
            }

            const item = currentResults.find((result) => result._id === id);
            if (!item || newName === item.food) {
                return;
            }

            await applyFoodNameChange(id, newName);
        };

        input.addEventListener("blur", trigger);
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                input.blur();
            }
        });
    });

    // Toggle for expanded nutrients
    document.querySelectorAll(".nutrients-toggle").forEach((button) => {
        button.addEventListener("click", () => {
            const id = button.dataset.id;
            const extraDiv = document.getElementById(`nutrients-extra-${id}`);
            const isHidden = extraDiv.style.display === 'none';
            extraDiv.style.display = isHidden ? 'block' : 'none';
            button.textContent = isHidden ? 'Hide Additional Nutrients' : 'Show All Nutrients (27 Total)';
        });
    });
}

function applyWeightChange(id, newWeight) {
    const item = currentResults.find((result) => result._id === id);
    if (!item) {
        return;
    }

    item.weight = newWeight;
    item.nutrients = scaleNutrients(item._nutrientsPer100g || {}, newWeight);
    patchNutrientCells(id, item.nutrients);
    renderSummary();
}

async function applyFoodNameChange(id, newName) {
    const item = currentResults.find((result) => result._id === id);
    if (!item) {
        return;
    }

    const card = document.querySelector(`.result-card[data-id="${id}"]`);
    card?.classList.add("card-loading");

    try {
        const formData = new FormData();
        formData.append("food", newName);
        formData.append("weight", String(item.weight));

        const response = await fetch("/nutrition", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok || payload.error) {
            throw new Error(payload.error || "Nutrition lookup failed");
        }

        item.food = newName;
        item.nutrients = payload.nutrients || {};
        item._nutrientsPer100g = computePer100g(item.nutrients, item.weight);

        const matchedElement = card?.querySelector(".matched-name");
        if (matchedElement) {
            matchedElement.textContent = item.nutrients.matched_name || "Not found";
        }

        patchNutrientCells(id, item.nutrients);
        renderSummary();
    } catch (error) {
        console.error("Nutrition lookup failed:", error);
        const nameInput = card?.querySelector(".food-name-input");
        if (nameInput) {
            nameInput.value = item.food || "";
        }
        setStatus(error.message, true);
    } finally {
        card?.classList.remove("card-loading");
    }
}

function patchNutrientCells(id, nutrients) {
    const nutrientsDiv = document.getElementById(`nutrients-${id}`);
    if (nutrientsDiv) {
        const { keyHtml, extraHtml } = createNutrientRows(nutrients);
        nutrientsDiv.innerHTML = keyHtml;
        
        // Update extra nutrients if they exist
        const extraDiv = document.getElementById(`nutrients-extra-${id}`);
        if (extraDiv && extraHtml) {
            extraDiv.innerHTML = `<div class="nutrients-divider">Additional Nutrients</div>${extraHtml}`;
        }
    }
}
