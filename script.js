// Function to simulate OTP verification and redirection
function validateOTP(event) {
  event.preventDefault(); // Prevent the default form submission behavior

  // Show loading animation
  const loading = document.getElementById("loading");
  loading.style.display = "block";

  // Simulate OTP verification process (e.g., backend call)
  setTimeout(() => {
      const phone = document.getElementById("phone").value;
      const otp = document.getElementById("otp").value;

      // Simulated OTP check (for demonstration)
      if (otp === "123456") {
          loading.textContent = "OTP Verified! Login Successful!";
          loading.style.color = "green";

          // Redirect to index3.html after successful login
          setTimeout(() => {
              window.location.href = "index3.html"; // Redirect to your target page
          }, 2000); // 2 seconds delay before redirect
      } else {
          loading.textContent = "Invalid OTP. Please try again.";
          loading.style.color = "red";
          setTimeout(() => {
              loading.style.display = "none"; // Hide loading after message
          }, 2000); // Hide message after 2 seconds
      }
  }, 2000); // 2 seconds delay for simulating OTP validation process
}
