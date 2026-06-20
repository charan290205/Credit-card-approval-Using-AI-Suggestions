document.getElementById('loginForm').addEventListener('submit', function(e) {
  e.preventDefault();

  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value.trim();

  if (!username || !password) {
    alert('Please fill in both fields');
    return;
  }

  // Demo success → redirect
  // In real app → send to your backend
  alert('Login successful! Redirecting to dashboard...');
  window.location.href = "http://localhost:8501";   // ← change this to your real Streamlit URL
});

// Password visibility toggle
const togglePassword = document.getElementById('togglePassword');
const passwordInput = document.getElementById('password');

togglePassword.addEventListener('click', function () {
  const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
  passwordInput.setAttribute('type', type);
  
  this.classList.toggle('fa-eye-slash');
  this.classList.toggle('fa-eye');
});