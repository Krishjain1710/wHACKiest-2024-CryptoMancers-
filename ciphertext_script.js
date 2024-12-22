document.getElementById('ciphertextForm').addEventListener('submit', (e) => {
    e.preventDefault();
  
    const ciphertext = document.getElementById('ciphertext').value;
  
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
  
    loadingDiv.style.display = 'block';
    resultDiv.textContent = '';
  
    setTimeout(() => {
      loadingDiv.style.display = 'none';
  
      // Simulated algorithm detection
      const algorithms = ['AES', 'RSA', 'SHA-256', 'DES', 'Blowfish'];
      const randomAlgorithm = algorithms[Math.floor(Math.random() * algorithms.length)];
  
      resultDiv.textContent = `Detected Algorithm: ${randomAlgorithm}`;
    }, 2000);
  });
  