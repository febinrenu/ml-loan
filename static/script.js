// API endpoint - change this when deploying
const API_URL = window.location.origin;

// Form submission handler
document.getElementById('loanForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        Name: document.getElementById('name').value,
        Gender: document.getElementById('gender').value,
        Married: document.getElementById('married').value,
        Dependents: document.getElementById('dependents').value,
        Education: document.getElementById('education').value,
        Self_Employed: document.getElementById('self_employed').value,
        ApplicantIncome: parseFloat(document.getElementById('applicant_income').value),
        CoapplicantIncome: parseFloat(document.getElementById('coapplicant_income').value),
        LoanAmount: parseFloat(document.getElementById('loan_amount').value),
        Loan_Amount_Term: parseFloat(document.getElementById('loan_term').value),
        Credit_History: parseFloat(document.getElementById('credit_history').value),
        Property_Area: document.getElementById('property_area').value
    };
    
    // Validate data
    if (!validateFormData(formData)) {
        return;
    }
    
    // Show loading state
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    
    submitBtn.disabled = true;
    btnText.textContent = 'Processing...';
    loader.style.display = 'inline-block';
    
    try {
        // Send prediction request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResult(result);
        } else {
            displayError(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        displayError('Failed to connect to the server. Please try again.');
    } finally {
        // Reset button
        submitBtn.disabled = false;
        btnText.textContent = 'Predict Eligibility';
        loader.style.display = 'none';
    }
});

// Validate form data
function validateFormData(data) {
    // Check for empty fields
    for (const [key, value] of Object.entries(data)) {
        if (value === '' || value === null || value === undefined) {
            displayError(`Please fill in all required fields: ${key}`);
            return false;
        }
    }
    
    // Validate numeric fields
    if (data.ApplicantIncome < 0 || data.CoapplicantIncome < 0 || data.LoanAmount <= 0) {
        displayError('Income and loan amount must be positive values');
        return false;
    }
    
    return true;
}

// Display prediction result
function displayResult(result) {
    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');
    const formContainer = document.querySelector('.form-container');
    
    const isApproved = result.prediction === 'Approved';
    const confidence = (result.confidence * 100).toFixed(2);
    const probabilityApproved = (result.probability_approved * 100).toFixed(2);
    const probabilityRejected = (result.probability_rejected * 100).toFixed(2);
    
    const resultClass = isApproved ? 'result-approved' : 'result-rejected';
    const icon = isApproved ? '✅' : '❌';
    const title = isApproved ? 'Loan Approved!' : 'Loan Rejected';
    const message = isApproved 
        ? `Congratulations ${result.applicant_name}! Your loan application has been approved.`
        : `Sorry ${result.applicant_name}, your loan application has been rejected. Please review your application details.`;
    
    resultContent.innerHTML = `
        <div class="${resultClass}">
            <div class="result-icon">${icon}</div>
            <div class="result-title">${title}</div>
            <div class="result-message">${message}</div>
            
            <div class="result-details">
                <div class="detail-row">
                    <span class="detail-label">Applicant:</span>
                    <span>${result.applicant_name}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Decision:</span>
                    <span><strong>${result.prediction}</strong></span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Confidence:</span>
                    <span><strong>${confidence}%</strong></span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Approval Probability:</span>
                    <span><strong>${probabilityApproved}%</strong></span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Rejection Probability:</span>
                    <span><strong>${probabilityRejected}%</strong></span>
                </div>
            </div>
            
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence}%;">
                    ${confidence}%
                </div>
            </div>
        </div>
    `;
    
    // Hide form and show result
    formContainer.style.display = 'none';
    resultContainer.style.display = 'block';
    
    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

// Display error message
function displayError(message) {
    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');
    const formContainer = document.querySelector('.form-container');
    
    resultContent.innerHTML = `
        <div class="error-message">
            <strong>Error:</strong> ${message}
        </div>
    `;
    
    formContainer.style.display = 'none';
    resultContainer.style.display = 'block';
}

// Reset form
function resetForm() {
    const formContainer = document.querySelector('.form-container');
    const resultContainer = document.getElementById('resultContainer');
    
    // Show form and hide result
    formContainer.style.display = 'block';
    resultContainer.style.display = 'none';
    
    // Scroll to form
    formContainer.scrollIntoView({ behavior: 'smooth' });
}

// Load model info on page load
window.addEventListener('load', async function() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const health = await response.json();
        
        if (!health.model_loaded || !health.preprocessor_loaded) {
            console.warn('Model or preprocessor not loaded properly');
        }
    } catch (error) {
        console.error('Failed to check server health:', error);
    }
});
