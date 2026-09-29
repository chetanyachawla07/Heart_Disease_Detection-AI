const form = document.getElementById("patient-form");
const resultBox = document.getElementById("result");
const resultLabel = document.getElementById("result-label");
const resultProbValue = document.getElementById("result-prob-value");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  const submitBtn = form.querySelector("button[type=submit]");
  submitBtn.textContent = "Assessing...";
  submitBtn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    resultBox.hidden = false;
    resultBox.classList.remove("risk", "safe");
    resultBox.classList.add(data.prediction === 1 ? "risk" : "safe");
    resultLabel.textContent = data.label;
    resultProbValue.textContent = data.probability;

    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    alert("Something went wrong: " + err.message);
  } finally {
    submitBtn.textContent = "Assess risk";
    submitBtn.disabled = false;
  }
});
