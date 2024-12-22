document.addEventListener("DOMContentLoaded", () => {
    const detectorForm = document.getElementById("detectorForm");
    const loading = document.getElementById("loading");
    const result = document.getElementById("result");
  
    detectorForm.addEventListener("submit", (e) => {
      e.preventDefault();
  
      // Clear previous results and show loading
      result.textContent = "";
      loading.style.display = "block";
  
      // Simulate backend processing
      setTimeout(() => {
        loading.style.display = "none";
  
        // Randomly select a cryptographic algorithm
        const algorithms = ["AES", "RSA", "SHA-256", "DES", "Blowfish"];
        const randomAlgorithm = algorithms[Math.floor(Math.random() * algorithms.length)];
  
        result.textContent = `Detected Algorithm: ${randomAlgorithm}`;
      }, 2000); // Simulate a 2-second delay
    });
  });
  