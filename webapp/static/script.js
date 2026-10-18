const form = document.getElementById("patient-form");
const resultBox = document.getElementById("result");
const submitBtn = form.querySelector("button[type=submit]");

const $ = (id) => document.getElementById(id);

function clearErrors() {
  form.querySelectorAll(".error").forEach((el) => (el.textContent = ""));
  form.querySelectorAll("[aria-invalid]").forEach((el) => el.removeAttribute("aria-invalid"));
}

function showFieldErrors(fields) {
  let first = null;
  for (const [name, msg] of Object.entries(fields)) {
    const err = $("err-" + name);
    const input = $("f-" + name);
    if (err) err.textContent = msg;
    if (input) {
      input.setAttribute("aria-invalid", "true");
      first = first || input;
    }
  }
  if (first) first.focus();
}

function buildPayload() {
  const payload = {};
  for (const [name, value] of new FormData(form).entries()) {
    payload[name] = value === "" ? "" : Number(value);
  }
  return payload;
}

function renderResult(data) {
  resultBox.hidden = false;
  resultBox.classList.remove("risk", "safe");
  resultBox.classList.add(data.flagged ? "risk" : "safe");

  $("result-label").textContent = data.label;
  $("result-prob-value").textContent = data.probability;
  $("meter-fill").style.width = data.probability + "%";
  $("meter-mark").style.left = data.threshold + "%";
  $("meter-note").textContent =
    "The flag is raised at " + data.threshold + "% (the tick on the bar), chosen to catch most real cases.";

  const warnings = $("warnings");
  warnings.replaceChildren(
    ...data.warnings.map((w) => Object.assign(document.createElement("li"), { textContent: w }))
  );

  const list = $("factors");
  list.replaceChildren();
  data.factors.forEach((f) => {
    const li = document.createElement("li");
    li.className = f.direction === "raises" ? "up" : "down";
    const name = document.createElement("span");
    name.textContent = f.label + ": " + f.value;
    const effect = document.createElement("strong");
    effect.textContent = (f.direction === "raises" ? "▲ raises" : "▼ lowers") +
      " odds ×" + f.odds_ratio.toFixed(2);
    li.append(name, effect);
    list.append(li);
  });
  $("factors-wrap").hidden = data.factors.length === 0;

  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();
  submitBtn.textContent = "Assessing...";
  submitBtn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const data = await res.json();

    if (!res.ok) {
      resultBox.hidden = true;
      if (data.fields) showFieldErrors(data.fields);
      else alert(data.error || "Something went wrong.");
      return;
    }
    renderResult(data);
  } catch (err) {
    alert("Could not reach the server: " + err.message);
  } finally {
    submitBtn.textContent = "Assess risk";
    submitBtn.disabled = false;
  }
});

$("reset-btn").addEventListener("click", () => {
  form.reset();
  clearErrors();
  resultBox.hidden = true;
});
