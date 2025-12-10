// API Configuration
const API_URL = 'http://localhost:5000';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const emptyState = document.getElementById('emptyState');
const resultsContainer = document.getElementById('resultsContainer');
const successAlert = document.getElementById('successAlert');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');

// Patient info elements
const patientName = document.getElementById('patientName');
const patientAge = document.getElementById('patientAge');
const patientSex = document.getElementById('patientSex');

// Chatbot variables
let currentMedicalAnalysis = null;
let currentConditionName = null;
let currentConfidenceScore = null;

// Report state
let lastReportData = null;

// Patient data
let currentPatientData = {
    name: '',
    age: '',
    sex: ''
};

// Check backend connection on load
window.addEventListener('load', () => {
    checkBackendHealth();
    setupPatientFormValidation();
});

async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            successAlert.classList.add('active');
            console.log('✓ Backend connected successfully');
        } else {
            showError('Backend is running but model not loaded');
        }
    } catch (error) {
        showError('Cannot connect to backend. Make sure Flask server is running on port 5000');
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorAlert.classList.add('active');
    successAlert.classList.remove('active');
}

function hideError() {
    errorAlert.classList.remove('active');
}

// Patient form validation
function setupPatientFormValidation() {
    const inputs = [patientName, patientAge, patientSex];
    
    inputs.forEach(input => {
        input.addEventListener('input', validatePatientForm);
        input.addEventListener('change', validatePatientForm);
    });
}

function validatePatientForm() {
    const name = patientName.value.trim();
    const age = patientAge.value.trim();
    const sex = patientSex.value;
    
    const isValid = name !== '' && age !== '' && sex !== '' && parseInt(age) > 0 && parseInt(age) <= 120;
    
    if (isValid) {
        uploadArea.classList.remove('disabled');
        fileInput.disabled = false;
        uploadArea.style.cursor = 'pointer';
        currentPatientData = { name, age, sex };
    } else {
        uploadArea.classList.add('disabled');
        fileInput.disabled = true;
        uploadArea.style.cursor = 'not-allowed';
    }
    
    return isValid;
}

// Click to upload
uploadArea.addEventListener('click', () => {
    if (!uploadArea.classList.contains('disabled')) {
        fileInput.click();
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (!uploadArea.classList.contains('disabled')) {
        uploadArea.classList.add('dragover');
    }
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    if (uploadArea.classList.contains('disabled')) {
        showError('Please fill in all patient information before uploading an image');
        return;
    }
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

async function handleFile(file) {
    // Validate patient form first
    if (!validatePatientForm()) {
        showError('Please fill in all patient information before uploading an image');
        return;
    }

    // Validate file
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG)');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    // Hide errors
    hideError();

    // Hide patient form and upload area, show preview
    document.getElementById('patientFormContainer').classList.add('hidden');
    previewContainer.classList.remove('hidden');

    // Update preview with patient info
    document.getElementById('previewPatientName').textContent = currentPatientData.name;
    document.getElementById('previewPatientDetails').textContent = 
        `${currentPatientData.age} years • ${currentPatientData.sex}`;

    // Show loading in results section
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    loading.classList.add('active');
    successAlert.classList.remove('active');

    // Read and display image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Send to backend for prediction
    await getPrediction(file);
}

async function getPrediction(file) {
    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', file);
        formData.append('patient_name', currentPatientData.name);
        formData.append('patient_age', currentPatientData.age);
        formData.append('patient_sex', currentPatientData.sex);

        // Send request to Flask backend
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
            loading.classList.remove('active');
            emptyState.classList.remove('hidden');
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showError('Failed to connect to server. Make sure Flask is running on port 5000');
        loading.classList.remove('active');
        emptyState.classList.remove('hidden');
    }
}

function displayResults(data) {
    loading.classList.remove('active');
    resultsContainer.classList.remove('hidden');

    // Get top prediction from response
    const topPrediction = data.top_prediction || data.predictions?.[0];
    
    if (!topPrediction) {
        showError('No prediction data received');
        return;
    }

    // Update prediction label with confidence score
    document.getElementById('predictionLabel').textContent = 
        `🎯 Predicted: ${topPrediction.label}`;

    // Display only top prediction with confidence score prominently
    const resultsHTML = `
        <div class="result-item" style="animation: slideIn 0.5s ease both; border-left-width: 8px;">
            <div class="result-label" style="font-size: 1.3em; margin-bottom: 15px;">
                <span style="font-weight: 700;">${topPrediction.label}</span>
                <span class="result-percentage" style="font-size: 1.4em; font-weight: 700;">${topPrediction.percentage.toFixed(2)}%</span>
            </div>
            <div class="confidence-bar" style="height: 35px;">
                <div class="confidence-fill" style="width: ${topPrediction.percentage}%"></div>
            </div>
            <div style="margin-top: 15px; color: #475569; font-size: 1em; font-weight: 600;">
                <strong>Confidence Score:</strong> ${(topPrediction.confidence * 100).toFixed(2)}%
                ${data.model_accuracy ? `<br><strong>Model Accuracy:</strong> ${data.model_accuracy}%` : ''}
            </div>
        </div>
    `;

    document.getElementById('confidenceResults').innerHTML = resultsHTML;

    // Display medical analysis from Gemini
    displayMedicalAnalysis(data.medical_analysis);

    // Display recommended doctors
    displayRecommendedDoctors(data.recommended_doctors);

    // Store medical analysis context for chatbot
    if (data.medical_analysis && data.medical_analysis.analysis) {
        currentMedicalAnalysis = data.medical_analysis.analysis;
        currentConditionName = topPrediction.label;
        currentConfidenceScore = topPrediction.confidence;
        
        // Show chatbot icon
        document.getElementById('chatbotIcon').classList.remove('hidden');
    }

    // Persist for report download
    lastReportData = {
        patientName: currentPatientData.name,
        patientAge: currentPatientData.age,
        patientSex: currentPatientData.sex,
        detectedCondition: topPrediction.label,
        confidencePercent: (topPrediction.confidence * 100).toFixed(2),
        modelAccuracy: data.model_accuracy,
        medicalAnalysis: data.medical_analysis?.analysis || '',
        doctors: (data.recommended_doctors && data.recommended_doctors.doctors) ? data.recommended_doctors.doctors : [],
        doctorsMeta: {
            totalFound: data.recommended_doctors?.total_found || 0,
            condition: data.recommended_doctors?.condition || topPrediction.label
        }
    };
}

function displayMedicalAnalysis(medicalAnalysis) {
    const analysisContainer = document.getElementById('medicalAnalysisContainer');
    const analysisContent = document.getElementById('medicalAnalysis');

    if (!medicalAnalysis) {
        analysisContainer.classList.add('hidden');
        return;
    }

    if (medicalAnalysis.error) {
        let errorMessage = medicalAnalysis.error;
        let suggestion = '';
        
        if (medicalAnalysis.suggestion) {
            suggestion = `<br><br><strong>💡 Suggestion:</strong> ${medicalAnalysis.suggestion}`;
        } else if (errorMessage.includes('Rate limit') || errorMessage.includes('quota')) {
            suggestion = `<br><br><strong>💡 Tips:</strong> 
                <ul style="margin-top: 10px; padding-left: 20px;">
                    <li>Wait 1-2 minutes before trying again</li>
                    <li>Free tier has strict rate limits (15 requests per minute)</li>
                    <li>Consider upgrading your Gemini API plan for higher limits</li>
                    <li>The system will automatically retry with delays</li>
                </ul>`;
        } else if (errorMessage.includes('API key')) {
            suggestion = `<br><br><small>Set your GEMINI_API_KEY environment variable or in .env file.</small>`;
        }
        
        analysisContent.innerHTML = `
            <div class="analysis-error">
                <strong>⚠️ Analysis Unavailable:</strong> ${errorMessage}
                ${suggestion}
            </div>
        `;
        analysisContainer.classList.remove('hidden');
        return;
    }

    if (medicalAnalysis.analysis) {
        let formattedAnalysis = medicalAnalysis.analysis;
        
        // Convert **bold** to <strong>
        formattedAnalysis = formattedAnalysis.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert numbered sections with bold headers to h3
        formattedAnalysis = formattedAnalysis.replace(/^(\d+\.\s+\*\*)(.*?)(\*\*)/gm, '<h3>$2</h3>');
        
        // Convert standalone bold text to headers
        formattedAnalysis = formattedAnalysis.replace(/^\*\*(.*?)\*\*$/gm, '<h3>$1</h3>');
        
        // Convert bullet points
        formattedAnalysis = formattedAnalysis.replace(/^[-•]\s+(.*)$/gm, '<li>$1</li>');
        
        // Wrap consecutive list items in <ul>
        formattedAnalysis = formattedAnalysis.replace(/(<li>.*?<\/li>(?:\s*<li>.*?<\/li>)*)/gs, '<ul>$1</ul>');
        
        // Split into paragraphs
        const lines = formattedAnalysis.split('\n');
        let result = [];
        let currentParagraph = [];
        
        lines.forEach(line => {
            const trimmed = line.trim();
            if (!trimmed) {
                if (currentParagraph.length > 0) {
                    result.push(`<p>${currentParagraph.join(' ')}</p>`);
                    currentParagraph = [];
                }
            } else if (trimmed.startsWith('<h3>') || trimmed.startsWith('<ul>') || trimmed.startsWith('<li>')) {
                if (currentParagraph.length > 0) {
                    result.push(`<p>${currentParagraph.join(' ')}</p>`);
                    currentParagraph = [];
                }
                result.push(trimmed);
            } else {
                currentParagraph.push(trimmed);
            }
        });
        
        if (currentParagraph.length > 0) {
            result.push(`<p>${currentParagraph.join(' ')}</p>`);
        }
        
        formattedAnalysis = result.join('\n');

        analysisContent.innerHTML = formattedAnalysis;
        analysisContainer.classList.remove('hidden');
    } else {
        analysisContainer.classList.add('hidden');
    }
}

function resetUpload() {
    fileInput.value = '';
    
    // Clear patient form
    patientName.value = '';
    patientAge.value = '';
    patientSex.value = '';
    currentPatientData = { name: '', age: '', sex: '' };
    
    // Reset left side
    document.getElementById('patientFormContainer').classList.remove('hidden');
    previewContainer.classList.add('hidden');
    uploadArea.classList.add('disabled');
    fileInput.disabled = true;
    
    // Reset right side
    resultsContainer.classList.add('hidden');
    loading.classList.remove('active');
    emptyState.classList.remove('hidden');
    document.getElementById('medicalAnalysisContainer').classList.add('hidden');
    
    hideError();
    successAlert.classList.add('active');
    
    // Hide chatbot and clear context
    document.getElementById('chatbotIcon').classList.add('hidden');
    document.getElementById('chatbotModal').classList.remove('active');
    document.getElementById('doctorsSection').classList.add('hidden');
    currentMedicalAnalysis = null;
    currentConditionName = null;
    currentConfidenceScore = null;
    document.getElementById('chatbotMessages').innerHTML = `
        <div class="chatbot-empty">
            <div class="chatbot-empty-icon">💬</div>
            <p>Ask me anything about your medical analysis!</p>
            <p style="font-size: 0.9em; margin-top: 10px; color: #666;">I can help explain the condition, treatments, symptoms, and more.</p>
        </div>
    `;
}

function displayRecommendedDoctors(doctorsData) {
    const doctorsSection = document.getElementById('doctorsSection');
    const doctorsResults = document.getElementById('doctorsResults');

    if (!doctorsData) {
        doctorsSection.classList.add('hidden');
        return;
    }

    if (doctorsData.error || !doctorsData.doctors || doctorsData.doctors.length === 0) {
        doctorsResults.innerHTML = `
            <div style="text-align: center; padding: 30px; color: #64748b;">
                <p style="font-size: 1.2em; font-weight: 600;">No doctors found</p>
                <p style="margin-top: 10px; font-size: 1em;">
                    ${doctorsData.error || 'No doctors available for this condition.'}
                </p>
            </div>
        `;
        doctorsSection.classList.remove('hidden');
        return;
    }

    let html = '';
    const cards = doctorsData.doctors;
    
    cards.forEach(doctor => {
        html += createDoctorCard(doctor);
    });

    doctorsResults.innerHTML = html;
    doctorsSection.classList.remove('hidden');
}

function createDoctorCard(doctor) {
    const rating = doctor.rating ? `⭐ ${doctor.rating.toFixed(1)}` : '';
    const reviews = doctor.total_reviews ? `(${doctor.total_reviews} reviews)` : '';
    const experience = doctor.experience_years ? `${doctor.experience_years} years experience` : '';
    const fee = doctor.consultation_fee_pkr ? `PKR ${doctor.consultation_fee_pkr}` : '';

    return `
        <div class="doctor-card">
            <div class="doctor-header">
                <div>
                    <div class="doctor-name">${escapeHtml(doctor.doctor_name)}</div>
                    <div class="doctor-specialization">${escapeHtml(doctor.specialization)}</div>
                </div>
                ${rating ? `
                    <div class="doctor-rating">
                        <span class="stars">${rating}</span>
                        <span>${reviews}</span>
                    </div>
                ` : ''}
            </div>
            
            <div class="doctor-info">
                <span class="doctor-info-icon">🏥</span>
                <span>${escapeHtml(doctor.hospital_name)}</span>
            </div>
            
            <div class="doctor-info">
                <span class="doctor-info-icon">📍</span>
                <span>${escapeHtml(doctor.address)}, ${escapeHtml(doctor.city)}, ${escapeHtml(doctor.province)}</span>
            </div>

            ${doctor.phone && doctor.phone !== 'Not available' ? `
                <div class="doctor-info">
                    <span class="doctor-info-icon">📞</span>
                    <span>${escapeHtml(doctor.phone)}</span>
                </div>
            ` : ''}

            <div class="doctor-details">
                ${experience ? `
                    <div class="doctor-detail-item">
                        <span>👨‍⚕️</span>
                        <span>${experience}</span>
                    </div>
                ` : ''}
                ${fee ? `
                    <div class="doctor-detail-item">
                        <span>💰</span>
                        <span>Fee: ${fee}</span>
                    </div>
                ` : ''}
                ${doctor.availability ? `
                    <div class="doctor-detail-item">
                        <span>🕐</span>
                        <span>${escapeHtml(doctor.availability)}</span>
                    </div>
                ` : ''}
            </div>

            ${doctor.review_summary ? `
                <div style="margin-top: 12px; padding: 12px; background: #fef3c7; border-radius: 10px; font-size: 0.95em; color: #92400e; font-weight: 500;">
                    ${escapeHtml(doctor.review_summary)}
                </div>
            ` : ''}

            <div class="doctor-actions">
                ${doctor.phone && doctor.phone !== 'Not available' ? `
                    <a href="tel:${doctor.phone.replace(/\s/g, '')}" class="doctor-action-btn call">📞 Call</a>
                ` : ''}
                ${doctor.email && doctor.email !== 'Not available' ? `
                    <a href="mailto:${doctor.email}" class="doctor-action-btn">✉️ Email</a>
                ` : ''}
            </div>
        </div>
    `;
}

// Chatbot Functions
function toggleChatbot() {
    const modal = document.getElementById('chatbotModal');
    if (!currentMedicalAnalysis) {
        alert('Please upload an image and get analysis results first before asking questions.');
        return;
    }
    modal.classList.toggle('active');
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chatbotInput');
    const sendBtn = document.getElementById('chatbotSend');
    const messagesContainer = document.getElementById('chatbotMessages');
    
    const question = input.value.trim();
    if (!question) return;

    if (!currentMedicalAnalysis) {
        alert('No medical analysis available. Please upload an image first.');
        return;
    }

    // Clear empty state
    if (messagesContainer.querySelector('.chatbot-empty')) {
        messagesContainer.innerHTML = '';
    }

    // Add user message
    addChatMessage(question, 'user');
    input.value = '';
    sendBtn.disabled = true;

    // Add loading indicator
    const loadingId = addChatLoading();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                medical_analysis: currentMedicalAnalysis,
                condition_name: currentConditionName,
                confidence_score: currentConfidenceScore
            })
        });

        const data = await response.json();

        // Remove loading indicator
        document.getElementById(loadingId).remove();

        if (data.success) {
            addChatMessage(data.answer, 'bot');
        } else {
            addChatMessage(`Sorry, I couldn't process your question. ${data.error || 'Please try again.'}`, 'bot');
        }
    } catch (error) {
        console.error('Chat error:', error);
        document.getElementById(loadingId).remove();
        addChatMessage('Sorry, there was an error connecting to the server. Please try again.', 'bot');
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

function addChatMessage(text, type) {
    const messagesContainer = document.getElementById('chatbotMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chatbot-message ${type}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let formattedText = escapeHtml(text);
    if (type === 'bot') {
        formattedText = formattedText.replace(/\n/g, '<br>');
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }
    
    messageDiv.innerHTML = `
        <div class="message-bubble">${formattedText}</div>
        <div class="message-time">${time}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addChatLoading() {
    const messagesContainer = document.getElementById('chatbotMessages');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chatbot-message bot';
    const loadingId = 'loading-' + Date.now();
    loadingDiv.id = loadingId;
    
    loadingDiv.innerHTML = `
        <div class="message-bubble chatbot-loading">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    messagesContainer.appendChild(loadingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return loadingId;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function downloadReport(format = 'pdf') {
    if (!lastReportData) {
        alert('No report available. Please upload an image and generate results first.');
        return;
    }
    
    const endpoint = `${API_URL}/download_report?format=${encodeURIComponent(format)}`;
    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(lastReportData)
    })
    .then(async (res) => {
        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Failed to generate report' }));
            throw new Error(err.error || 'Failed to generate report');
        }
        return res.blob();
    })
    .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const ext = format === 'docx' ? 'docx' : 'pdf';
        const safeName = (lastReportData.patientName || 'Patient').replace(/[^a-z0-9]+/gi, '_');
        const safeCond = (lastReportData.detectedCondition || 'condition').replace(/[^a-z0-9]+/gi, '_');
        const ts = new Date().toISOString().replace(/[:T]/g, '-').split('.')[0];
        a.download = `Medical_Report_${safeName}_${safeCond}_${ts}.${ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    })
    .catch((err) => {
        alert(err.message || 'Report download failed');
    });
}