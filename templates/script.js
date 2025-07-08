// Initialize EmailJS with your PUBLIC KEY
emailjs.init("TiAWPaSGackJ4Tnjz"); // Example: TiAWPaSGackJ4Tnjz

document.getElementById('contact-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const button = document.getElementById('send-button');
  button.value = "Sending...";

  emailjs.sendForm('service_sx7p59c', 'template_50tsbcj', this)
    .then(() => {
      alert("Email sent successfully!");
      button.value = "Send Email";
      this.reset();
    })
    .catch((err) => {
      alert("Failed to send email.\n" + JSON.stringify(err));
      button.value = "Send Email";
    });
});
