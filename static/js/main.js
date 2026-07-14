document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeIcon = document.getElementById('themeIcon');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const detectBtn = document.getElementById('detectBtn');
    const resetBtn = document.getElementById('resetBtn');
    const actionButtons = document.getElementById('actionButtons');
    const scanOverlay = document.getElementById('scanOverlay');
    const errorMessage = document.getElementById('errorMessage');
    
    // Result Elements
    const resultsPanel = document.getElementById('resultsPanel');
    const originalImg = document.getElementById('originalImg');
    const predictedImg = document.getElementById('predictedImg');
    
    // Stat Card Elements
    const valTotalFaults = document.getElementById('valTotalFaults');
    const valBroken = document.getElementById('valBroken');
    const valPollution = document.getElementById('valPollution');
    const valNormal = document.getElementById('valNormal');
    
    // Metadata Details
    const metaTime = document.getElementById('metaTime');
    const metaDevice = document.getElementById('metaDevice');
    const metaResolution = document.getElementById('metaResolution');
    const metaObjects = document.getElementById('metaObjects');
    
    const predictionTableBody = document.querySelector('#predictionTable tbody');
    const downloadBtn = document.getElementById('downloadBtn');
    const uploadAnotherBtn = document.getElementById('uploadAnotherBtn');

    let selectedFile = null;

    // --- Theme Controller ---
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        if (theme === 'light') {
            themeIcon.className = 'fas fa-moon';
        } else {
            themeIcon.className = 'fas fa-sun';
        }
    }

    // --- Drag & Drop Upload Events ---
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // --- File Handling & Validation ---
    function handleFile(file) {
        errorMessage.classList.add('d-none');
        errorMessage.textContent = '';
        
        // Allowed formats
        const allowedExtensions = ['jpg', 'jpeg', 'png', 'bmp'];
        const extension = file.name.split('.').pop().toLowerCase();
        
        if (!allowedExtensions.includes(extension)) {
            showError(`Invalid file format. Allowed formats: ${allowedExtensions.join(', ').toUpperCase()}`);
            return;
        }

        // 20 MB size limit
        const maxSize = 20 * 1024 * 1024;
        if (file.size > maxSize) {
            showError("File size exceeds 20 MB limit. Please select a smaller image.");
            return;
        }

        selectedFile = file;
        
        // Instant Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add('d-none');
            previewContainer.classList.remove('d-none');
            actionButtons.classList.remove('d-none');
        };
        reader.readAsDataURL(file);
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
        // Scroll to error
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // --- Reset Controller ---
    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('d-none');
        uploadPrompt.classList.remove('d-none');
        actionButtons.classList.add('d-none');
        errorMessage.classList.add('d-none');
        errorMessage.textContent = '';
        resultsPanel.classList.add('d-none');
    }

    resetBtn.addEventListener('click', resetUpload);
    uploadAnotherBtn.addEventListener('click', () => {
        resetUpload();
        // Smooth scroll back to top upload zone
        dropzone.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    // --- Detect / Run Inference ---
    detectBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        // Show scanning animation, lock controls
        scanOverlay.style.display = 'flex';
        detectBtn.disabled = true;
        resetBtn.disabled = true;
        errorMessage.classList.add('d-none');

        const formData = new FormData();
        formData.append('file', selectedFile);

        fetch('/detect', {
            method: 'POST',
            headers: {
                'Accept': 'application/json'
            },
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || "Inference failed.") });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                renderResults(data);
            } else {
                throw new Error(data.error || "Failed running prediction.");
            }
        })
        .catch(err => {
            console.error(err);
            showError(err.message || "An unexpected error occurred during prediction.");
        })
        .finally(() => {
            // Remove overlay and enable buttons
            scanOverlay.style.display = 'none';
            detectBtn.disabled = false;
            resetBtn.disabled = false;
        });
    });

    // --- Render Results ---
    function renderResults(data) {
        // Set images
        originalImg.src = data.original_url;
        predictedImg.src = data.predicted_url;
        
        // Update stats with nice numerical animation
        animateNumber(valTotalFaults, data.summary["Total Faults"]);
        animateNumber(valBroken, data.summary["Broken Insulator"]);
        animateNumber(valPollution, data.summary["Pollution Flashover"]);
        animateNumber(valNormal, data.summary["Normal Insulator"]);
        
        // Update metadata card
        metaTime.textContent = `${data.detection_time_ms} ms`;
        metaDevice.textContent = data.device.toUpperCase();
        metaResolution.textContent = data.resolution;
        metaObjects.textContent = data.predictions.length;
        
        // Build table
        predictionTableBody.innerHTML = '';
        if (data.predictions.length === 0) {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td colspan="4" class="text-center text-muted">No objects detected.</td>`;
            predictionTableBody.appendChild(tr);
        } else {
            data.predictions.forEach((pred, index) => {
                const tr = document.createElement('tr');
                
                // Class badge mapping
                let badgeClass = 'badge-normal';
                if (pred.class_name === "Broken Insulator") {
                    badgeClass = 'badge-broken';
                } else if (pred.class_name === "Pollution Flashover") {
                    badgeClass = 'badge-pollution';
                }
                
                // Format Bounding Box
                const [x1, y1, x2, y2] = pred.bbox.map(val => Math.round(val));
                const bboxStr = `[${x1}, ${y1}, ${x2}, ${y2}]`;
                
                tr.innerHTML = `
                    <td class="fw-bold">${index + 1}</td>
                    <td><span class="badge-fault ${badgeClass}">${pred.class_name}</span></td>
                    <td class="fw-semibold">${(pred.confidence * 100).toFixed(1)}%</td>
                    <td class="font-monospace text-secondary" style="font-size: 0.85rem;">${bboxStr}</td>
                `;
                predictionTableBody.appendChild(tr);
            });
        }
        
        // Setup download button
        downloadBtn.onclick = () => {
            window.location.href = `/download/${data.filename}`;
        };
        
        // Slide in results panel
        resultsPanel.classList.remove('d-none');
        
        // Smooth scroll to predictions panel
        setTimeout(() => {
            resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
    }

    // Number counting animator helper
    function animateNumber(element, target) {
        let current = 0;
        const duration = 800; // ms
        const steps = 20;
        const stepTime = duration / steps;
        const increment = target / steps;
        
        element.textContent = '0';
        
        const interval = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target;
                clearInterval(interval);
            } else {
                element.textContent = Math.floor(current);
            }
        }, stepTime);
    }
});
