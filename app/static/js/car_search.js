const cars = {
  Toyota: ["Corolla", "Yaris", "Hilux"],
  BMW: ["320i", "X5", "M3"],
  Peugeot: ["208", "301", "508"]
};

function updateModels() {
  const brand = document.getElementById("brand").value;
  const modelSelect = document.getElementById("model");

  modelSelect.innerHTML = "";

  cars[brand].forEach(m => {
    let opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  });
}
