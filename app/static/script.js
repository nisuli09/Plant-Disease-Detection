const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");

let selectedFile = null;

imageInput.addEventListener("change", () => {
    selectedFile = imageInput.files[0];

    if (!selectedFile) {
        predictBtn.disabled = true;
        preview.classList.add("hidden");
        return;
    }

    preview.src = URL.createObjectURL(selectedFile);
    preview.classList.remove("hidden");
    predictBtn.disabled = false;
    result.classList.add("hidden");
});

predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        result.classList.remove("hidden");

        if (data.error) {
            result.textContent = data.error;
        } else {
            result.innerHTML =
                `<strong>Prediction:</strong> ${data.prediction}<br>` +
                `<strong>Confidence:</strong> ${data.confidence}%`;
        }
    } catch (error) {
        result.classList.remove("hidden");
        result.textContent = "Something went wrong. Please try again.";
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict";
    }
});
