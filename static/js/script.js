document.addEventListener('DOMContentLoaded', function() {
    // Update navigation handling to work with the side nav
    const navItems = document.querySelectorAll('.side-nav li');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Set up tab navigation - make sure this works with the new structure
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            // Remove active class from all nav items
            navItems.forEach(nav => nav.classList.remove('active'));
            
            // Add active class to clicked nav item
            this.classList.add('active');
            
            // Hide all tab contents
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            // Show the selected tab content
            const tabId = this.getAttribute('data-tab') + '-tab';
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Form elements
    const watermarkForm = document.getElementById('watermarkForm');
    const fileInput = document.getElementById('file');
    const fileName = document.getElementById('file-name');
    
    // Image watermark elements
    const watermarkImageInput = document.getElementById('watermark_image');
    const watermarkImageName = document.getElementById('watermark-image-name');
    const sizeSlider = document.getElementById('watermark_size');
    const sizeValue = document.getElementById('size-value');
    
    // Visibility elements
    const steganographyToggle = document.getElementById('steganography_toggle');
    
    // Process panels
    const progressContainer = document.getElementById('progress-container');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const downloadLink = document.getElementById('download-link');
    const errorMessage = document.getElementById('error-message');
    
    // Extraction elements
    const extractForm = document.getElementById('extractForm');
    const extractFileInput = document.getElementById('extract-file');
    const extractFileName = document.getElementById('extract-file-name');
    const extractResult = document.getElementById('extract-result');
    const extractedText = document.querySelector('.extracted-text');
    
    // Buttons
    const newWatermarkBtn = document.getElementById('new-watermark');
    const tryAgainBtn = document.getElementById('try-again');
    
    // Update selectors for progress, result, and error elements
    const createProgressSection = document.getElementById('create-progress');
    const createResultSection = document.getElementById('create-result');
    const createErrorSection = document.getElementById('create-error');
    const createProgressBar = document.getElementById('create-progress-bar');
    const createProgressText = document.getElementById('create-progress-text');
    const createErrorText = document.getElementById('create-error-text');
    const createResultPreview = document.getElementById('create-result-preview');
    
    const extractProgressSection = document.getElementById('extract-progress');
    const extractResultSection = document.getElementById('extract-result');
    const extractErrorSection = document.getElementById('extract-error');
    const extractErrorText = document.getElementById('extract-error-text');
    
    const createTryAgainBtn = document.getElementById('create-try-again');
    const extractNewBtn = document.getElementById('extract-new');
    const extractTryAgainBtn = document.getElementById('extract-try-again');

    // Add detection form elements
    const detectForm = document.getElementById('detectForm');
    const detectFileInput = document.getElementById('detect-file');
    const detectFileName = document.getElementById('detect-file-name');
    const detectProgressSection = document.getElementById('detect-progress');
    const detectResultSection = document.getElementById('detect-result');
    const detectErrorSection = document.getElementById('detect-error');
    const detectErrorText = document.getElementById('detect-error-text');
    const detectNewBtn = document.getElementById('detect-new');
    const detectTryAgainBtn = document.getElementById('detect-try-again');
    
    // Handle file selection and preview
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            fileName.textContent = file.name;
            
            // Get the preview container
            const previewContainer = document.querySelector('.media-preview');
            previewContainer.innerHTML = ''; // Clear previous content
            
            // Create and add loading indicator
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading-indicator';
            loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            previewContainer.appendChild(loadingIndicator);
            
            // Handle different file types
            if (file.type.startsWith('image/')) {
                // Handle image files
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.onload = function() {
                    URL.revokeObjectURL(this.src);
                    previewContainer.innerHTML = ''; // Remove loading indicator
                    previewContainer.appendChild(img);
                };
            } else if (file.type.startsWith('video/')) {
                // Handle video files
                const video = document.createElement('video');
                video.src = URL.createObjectURL(file);
                video.controls = true;
                video.muted = true;
                video.onloadeddata = function() {
                    previewContainer.innerHTML = ''; // Remove loading indicator
                    previewContainer.appendChild(video);
                    
                    // Add play/pause controls
                    const controls = document.createElement('div');
                    controls.className = 'preview-controls';
                    
                    const playBtn = document.createElement('button');
                    playBtn.innerHTML = '<i class="fas fa-play"></i>';
                    playBtn.addEventListener('click', () => video.play());
                    
                    const pauseBtn = document.createElement('button');
                    pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                    pauseBtn.addEventListener('click', () => video.pause());
                    
                    controls.appendChild(playBtn);
                    controls.appendChild(pauseBtn);
                    previewContainer.parentElement.appendChild(controls);
                };
            } else {
                // Unsupported file type
                previewContainer.innerHTML = '<div class="placeholder">Preview not available for this file type</div>';
            }
        } else {
            fileName.textContent = 'No file selected';
            document.querySelector('.media-preview').innerHTML = '<div class="placeholder">Preview will appear here</div>';
            
            // Remove any controls if they exist
            const controls = document.querySelector('.preview-controls');
            if (controls) controls.remove();
        }
    });
    
    // Handle extract file selection and preview
    extractFileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            extractFileName.textContent = file.name;
            
            // Get the preview container
            const previewContainer = document.getElementById('extract-media-preview');
            previewContainer.innerHTML = ''; // Clear previous content
            
            // Create and add loading indicator
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading-indicator';
            loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            previewContainer.appendChild(loadingIndicator);
            
            // Handle different file types
            if (file.type.startsWith('image/')) {
                // Handle image files
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.onload = function() {
                    URL.revokeObjectURL(this.src);
                    previewContainer.innerHTML = ''; // Remove loading indicator
                    previewContainer.appendChild(img);
                };
            } else if (file.type.startsWith('video/')) {
                // Handle video files
                const video = document.createElement('video');
                video.src = URL.createObjectURL(file);
                video.controls = true;
                video.muted = true;
                video.onloadeddata = function() {
                    previewContainer.innerHTML = ''; // Remove loading indicator
                    previewContainer.appendChild(video);
                    
                    // Add play/pause controls
                    const controls = document.createElement('div');
                    controls.className = 'preview-controls';
                    
                    const playBtn = document.createElement('button');
                    playBtn.innerHTML = '<i class="fas fa-play"></i>';
                    playBtn.addEventListener('click', () => video.play());
                    
                    const pauseBtn = document.createElement('button');
                    pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                    pauseBtn.addEventListener('click', () => video.pause());
                    
                    controls.appendChild(playBtn);
                    controls.appendChild(pauseBtn);
                    previewContainer.parentElement.appendChild(controls);
                };
            } else {
                // Unsupported file type
                previewContainer.innerHTML = '<div class="placeholder">Preview not available for this file type</div>';
            }
        } else {
            extractFileName.textContent = 'No file selected';
            document.getElementById('extract-media-preview').innerHTML = '<div class="placeholder">Preview will appear here</div>';
            
            // Remove any controls if they exist
            const controls = document.querySelector('.extract-media-preview-container .preview-controls');
            if (controls) controls.remove();
        }
    });

    // Handle detect file selection and preview
    detectFileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            detectFileName.textContent = file.name;
            
            // Get the preview container
            const previewContainer = document.getElementById('detect-media-preview');
            previewContainer.innerHTML = ''; // Clear previous content
            
            // Create and add loading indicator
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading-indicator';
            loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            previewContainer.appendChild(loadingIndicator);
            
            // Handle different file types
            if (file.type.startsWith('image/')) {
                // Handle image files
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.onload = function() {
                    URL.revokeObjectURL(this.src);
                    previewContainer.innerHTML = ''; // Remove loading indicator
                    previewContainer.appendChild(img);
                };
            } else if (file.type.startsWith('video/')) {
                // Handle video files
                const video = document.createElement('video');
                video.src = URL.createObjectURL(file);
                video.controls = true;
                video.muted = true;
                video.onloadeddata = function() {
                    previewContainer.innerHTML = ''; // Remove loading indicator
                    previewContainer.appendChild(video);
                    
                    // Add play/pause controls
                    const controls = document.createElement('div');
                    controls.className = 'preview-controls';
                    
                    const playBtn = document.createElement('button');
                    playBtn.innerHTML = '<i class="fas fa-play"></i>';
                    playBtn.addEventListener('click', () => video.play());
                    
                    const pauseBtn = document.createElement('button');
                    pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                    pauseBtn.addEventListener('click', () => video.pause());
                    
                    controls.appendChild(playBtn);
                    controls.appendChild(pauseBtn);
                    previewContainer.parentElement.appendChild(controls);
                };
            } else {
                // Unsupported file type
                previewContainer.innerHTML = '<div class="placeholder">Preview not available for this file type</div>';
            }
        } else {
            detectFileName.textContent = 'No file selected';
            document.getElementById('detect-media-preview').innerHTML = '<div class="placeholder">Preview will appear here</div>';
            
            // Remove any controls if they exist
            const controls = document.querySelector('#detect-media-preview-container .preview-controls');
            if (controls) controls.remove();
        }
    });

    // Fix watermark image selection and preview
    const watermarkPreview = document.querySelector('.watermark-preview');
    
    if (watermarkImageInput) {
        watermarkImageInput.addEventListener('change', function(e) {
            console.log('Watermark image selected:', this.files);
            
            if (this.files && this.files.length > 0) {
                const file = this.files[0];
                
                // Update file name display
                watermarkImageName.textContent = file.name;
                
                // Clear previous preview
                watermarkPreview.innerHTML = '';
                
                // Show loading indicators
                const loading = document.createElement('div');
                loading.className = 'loading-indicator';
                loading.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading preview...';
                watermarkPreview.appendChild(loading);
                
                // Create new image preview
                const img = new Image();
                const objectUrl = URL.createObjectURL(file);
                
                img.onload = function() {
                    console.log('Watermark image loaded successfully');
                    watermarkPreview.innerHTML = '';
                    watermarkPreview.appendChild(img);
                    URL.revokeObjectURL(objectUrl);
                };
                
                img.onerror = function() {
                    console.error('Failed to load watermark image preview');
                    watermarkPreview.innerHTML = '<div class="error-message">Failed to load preview</div>';
                    URL.revokeObjectURL(objectUrl);
                };
                
                img.className = 'watermark-preview-image';
                img.src = objectUrl;
            } else {
                watermarkImageName.textContent = 'No image selected';
                watermarkPreview.innerHTML = '<div class="placeholder">Preview will appear here</div>';
            }
        });
    } else {
        console.error('Watermark image input element not found');
    }
    
    // Handle size slider
    sizeSlider.addEventListener('input', function() {
        sizeValue.textContent = this.value + '%';
        // Uncheck all radio buttons when slider is manually adjusted
        document.querySelectorAll('input[name="size_preset"]').forEach(radio => {
            radio.checked = false;
        });
    });
    
    // Add opacity slider handler
    const opacitySlider = document.getElementById('watermark_opacity');
    const opacityValue = document.getElementById('opacity-value');
    
    if (opacitySlider) {
        opacitySlider.addEventListener('input', function() {
            opacityValue.textContent = this.value + '%';
            // Uncheck all radio buttons when slider is manually adjusted
            document.querySelectorAll('input[name="opacity_preset"]').forEach(radio => {
                radio.checked = false;
            });
        });
    }
    
    // Add handlers for size preset radio buttons
    document.querySelectorAll('input[name="size_preset"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                // Update the slider to match the selected preset
                sizeSlider.value = this.value;
                sizeValue.textContent = this.value + '%';
            }
        });
    });
    
    // Add handlers for opacity preset radio buttons
    document.querySelectorAll('input[name="opacity_preset"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                // Update the slider to match the selected preset
                opacitySlider.value = this.value;
                opacityValue.textContent = this.value + '%';
            }
        });
    });

    // Fix the watermarkForm submit handler
    watermarkForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Form validation
        if (fileInput.files.length === 0) {
            showCreateError('Please select a file to watermark');
            return;
        }
        
        const formData = new FormData(this);
        
        // Set watermark technique to combined as it's the only option
        formData.set('watermark_technique', 'combined');
        
        // Set to always use image watermark now that text is removed
        formData.set('content_type', 'image');
        
        // Add the opacity value to the form data
        const opacity = document.getElementById('watermark_opacity').value;
        formData.set('watermark_opacity', opacity);
        
        // Image watermark validation
        if (watermarkImageInput.files.length === 0) {
            showCreateError('Please select a watermark image');
            return;
        }
        
        // Set watermark type based on visibility toggle
        formData.set('watermark_type', steganographyToggle.checked ? 'steganographic' : 'visible');
        
        // Show progress section, hide form, result, and error sections
        document.querySelector('#watermarkForm').classList.add('hidden');
        createProgressSection.classList.remove('hidden');
        createResultSection.classList.add('hidden');
        createErrorSection.classList.add('hidden');
        
        // Reset progress
        createProgressBar.style.width = '0%';
        createProgressText.textContent = '0%';
        
        // Submit the data
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showCreateError(data.error);
                return;
            }
            
            // Start polling for progress
            const taskId = data.task_id;
            pollCreateProgress(taskId);
        })
        .catch(error => {
            showCreateError('Failed to connect to server: ' + error.message);
        });
    });

    // Enhanced poll progress for watermarking
    function pollCreateProgress(taskId) {
        fetch(`/progress/${taskId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showCreateError(data.error);
                return;
            }
            
            // Update progress bar
            createProgressBar.style.width = `${data.progress}%`;
            createProgressText.textContent = `${data.progress}%`;
            
            if (data.status === 'error') {
                showCreateError(data.message || 'An error occurred during processing');
            } else if (data.status === 'complete') {
                // Show success section
                downloadLink.href = `/download/${taskId}`;
                createProgressSection.classList.add('hidden');
                createResultSection.classList.remove('hidden');
                
                // Load the preview of the result
                showCreateResultPreview(taskId);
            } else {
                // Continue polling
                setTimeout(() => {
                    pollCreateProgress(taskId);
                }, 500);
            }
        })
        .catch(error => {
            showCreateError('Failed to check progress: ' + error.message);
        });
    }

    // Function to show the create result preview
    function showCreateResultPreview(taskId) {
        createResultPreview.innerHTML = '<div class="loading-indicator"><i class="fas fa-spinner fa-spin"></i></div>';
        
        // Get the preview URL
        fetch(`/preview/${taskId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                createResultPreview.innerHTML = '<div class="placeholder">Preview not available</div>';
                return;
            }
            
            // Check if it's an image or video
            if (data.mime_type.startsWith('image/')) {
                // It's an image
                const img = document.createElement('img');
                img.src = data.preview_url;
                img.onload = function() {
                    createResultPreview.innerHTML = '';
                    createResultPreview.appendChild(img);
                    
                    // Set the download link and don't specify a filename
                    // Let the server set it through Content-Disposition
                    downloadLink.href = `/download/${taskId}`;
                    
                    // Don't set download attribute so browser uses original filename
                    downloadLink.removeAttribute('download');
                };
            } else if (data.mime_type.startsWith('video/')) {
                // Handle video files
                const video = document.createElement('video');
                video.src = data.preview_url;
                video.controls = true;
                video.muted = true;
                video.onloadeddata = function() {
                    createResultPreview.innerHTML = '';
                    createResultPreview.appendChild(video);
                    
                    // Set the download link and don't specify a filename
                    downloadLink.href = `/download/${taskId}`;
                    
                    // Don't set download attribute to allow original filename
                    downloadLink.removeAttribute('download');
                    
                    // Add play/pause controls
                    const controls = document.createElement('div');
                    controls.className = 'preview-controls';
                    
                    const playBtn = document.createElement('button');
                    playBtn.innerHTML = '<i class="fas fa-play"></i>';
                    playBtn.addEventListener('click', () => video.play());
                    
                    const pauseBtn = document.createElement('button');
                    pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                    pauseBtn.addEventListener('click', () => video.pause());
                    
                    controls.appendChild(playBtn);
                    controls.appendChild(pauseBtn);
                    createResultPreview.parentElement.appendChild(controls);
                };
            } else {
                createResultPreview.innerHTML = '<div class="placeholder">Preview not available for this file type</div>';
            }
        })
        .catch(error => {
            createResultPreview.innerHTML = '<div class="placeholder">Error loading preview</div>';
        });
    }

    // Function to display create error
    function showCreateError(message) {
        createErrorText.textContent = message;
        createProgressSection.classList.add('hidden');
        createErrorSection.classList.remove('hidden');
        document.querySelector('#watermarkForm').classList.add('hidden');
    }

    // Function to display the extraction results - ENHANCED VERSION
    function showExtractionResults(data) {
        // Get the extracted watermark content container
        const extractedContent = document.getElementById('extracted-watermark-content');
        
        // Clear previous content
        extractedContent.innerHTML = '';
        
        console.log("Extraction result:", data); // Debug log
        
        // Check if we have extracted watermark image
        if (data.has_watermark_image && data.watermark_image_url) {
            // Create image element
            const img = document.createElement('img');
            img.src = data.watermark_image_url;
            img.alt = "Extracted Watermark Image";
            img.className = "extracted-watermark-image";
            
            // Create download link with no specific download attribute
            // Let the server control the filename via Content-Disposition
            const downloadLink = document.createElement('div');
            downloadLink.className = 'watermark-download';
            downloadLink.innerHTML = `<a href="${data.watermark_image_url}" class="btn btn-sm btn-secondary">
                <i class="fas fa-download"></i> Download Watermark Image
            </a>`;
            
            // Add elements to container
            extractedContent.appendChild(img);
            extractedContent.appendChild(downloadLink);
        } else {
            // If no watermark image, display the text watermark
            const textEl = document.createElement('div');
            textEl.className = "text-content";
            textEl.textContent = data.watermark || "No watermark found";
            extractedContent.appendChild(textEl);
        }
        
        // Show results
        extractResultSection.classList.remove('hidden');
    }

    // Updated extraction handler to focus on image watermarks
    extractForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (extractFileInput.files.length === 0) {
            showExtractError('Please select a file to extract the watermark from');
            return;
        }
        
        const formData = new FormData(extractForm);
        
        // Show progress section, hide form and result
        extractForm.classList.add('hidden');
        extractProgressSection.classList.remove('hidden');
        extractResultSection.classList.add('hidden');
        extractErrorSection.classList.add('hidden');
        
        // Submit the extraction request
        fetch('/extract', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            extractProgressSection.classList.add('hidden');
            
            if (data.status === 'error') {
                showExtractError(data.message || 'Failed to extract watermark');
                return;
            }
            
            // Display the extraction results
            const extractedContent = document.getElementById('extracted-watermark-content');
            extractedContent.innerHTML = '';
            
            if (data.has_watermark_image && data.watermark_image_url) {
                // Create image element
                const img = document.createElement('img');
                img.src = data.watermark_image_url;
                img.alt = "Extracted Watermark Image";
                img.className = "extracted-watermark-image";
                
                // Create download link
                const downloadLink = document.createElement('div');
                downloadLink.className = 'watermark-download';
                downloadLink.innerHTML = `<a href="${data.watermark_image_url}" class="btn btn-sm btn-secondary">
                    <i class="fas fa-download"></i> Download Watermark Image
                </a>`;
                
                // Add elements to container
                extractedContent.appendChild(img);
                extractedContent.appendChild(downloadLink);
            } else {
                // No watermark found
                const message = document.createElement('div');
                message.className = "text-content";
                message.textContent = data.watermark || "No image watermark was found";
                extractedContent.appendChild(message);
            }
            
            // Show results section
            extractResultSection.classList.remove('hidden');
        })
        .catch(error => {
            showExtractError('Failed to extract watermark: ' + error.message);
        });
    });

    // Function to display extract error
    function showExtractError(message) {
        extractErrorText.textContent = message;
        extractProgressSection.classList.add('hidden');
        extractErrorSection.classList.remove('hidden');
        extractForm.classList.add('hidden');
    }

    // Handle detection form submission
    detectForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (detectFileInput.files.length === 0) {
            showDetectError('Please select a file to detect watermarks');
            return;
        }
        
        const formData = new FormData(detectForm);
        
        // Show progress section, hide form and result
        detectForm.classList.add('hidden');
        detectProgressSection.classList.remove('hidden');
        detectResultSection.classList.add('hidden');
        detectErrorSection.classList.add('hidden');
        
        // Submit the detection request - change URL to new endpoint
        fetch('/attack', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            detectProgressSection.classList.add('hidden');
            
            if (data.status === 'error') {
                showDetectError(data.message || 'Failed to detect watermark');
                return;
            }
            
            // Display detection results
            showDetectionResults(data);
        })
        .catch(error => {
            showDetectError('Failed to detect watermark: ' + error.message);
        });
    });

    // Enhance the showDetectionResults function to add extraction capability
    function showDetectionResults(data) {
        // Display the original image
        const originalPreview = document.getElementById('detect-original-preview');
        originalPreview.innerHTML = '<div class="loading-indicator"><i class="fas fa-spinner fa-spin"></i></div>';
        
        const originalImg = document.createElement('img');
        originalImg.src = data.original_image_url;
        originalImg.onload = function() {
            originalPreview.innerHTML = '';
            originalPreview.appendChild(originalImg);
        };
        originalImg.onerror = function() {
            originalPreview.innerHTML = '<div class="placeholder">Failed to load image</div>';
        };
        
        // Display the attacked image
        const attackedPreview = document.getElementById('detect-attacked-preview');
        attackedPreview.innerHTML = '<div class="loading-indicator"><i class="fas fa-spinner fa-spin"></i></div>';
        
        const attackedImg = document.createElement('img');
        attackedImg.src = data.attacked_image_url;
        attackedImg.onload = function() {
            attackedPreview.innerHTML = '';
            attackedPreview.appendChild(attackedImg);
        };
        attackedImg.onerror = function() {
            attackedPreview.innerHTML = '<div class="placeholder">Failed to load image</div>';
        };
        
        // Update attack description
        document.getElementById('attack-description').textContent = `After ${data.attack_type}`;
        
        // Set download link for the attacked image with proper attributes
        const downloadLink = document.getElementById('download-detect-attacked-link');
        downloadLink.href = data.attacked_image_url;
        downloadLink.setAttribute('download', ''); // Force download behavior
        
        // Add click handler to ensure download works
        downloadLink.onclick = function() {
            // Let the browser handle the download and prompt for filename
            return true;
        };
        
        // Add event listener for "Extract from Attacked" button
        const extractFromAttackedBtn = document.getElementById('extract-from-attacked');
        extractFromAttackedBtn.onclick = function() {
            extractWatermarkFromAttacked(data.attacked_image_url);
        };
        
        // Show results
        detectResultSection.classList.remove('hidden');
    }

    // Add function to extract watermark from attacked image
    function extractWatermarkFromAttacked(imageUrl) {
        // Show loading state
        const extractionResult = document.getElementById('attack-extraction-result');
        extractionResult.classList.remove('hidden');
        extractionResult.innerHTML = `
            <h4><i class="fas fa-spinner fa-spin"></i> Extracting Watermark...</h4>
            <p>Attempting to extract watermark from the attacked image...</p>
        `;
        
        // Fetch the image as blob
        fetch(imageUrl)
            .then(response => response.blob())
            .then(blob => {
                // Create a FormData object to submit the image for extraction
                const formData = new FormData();
                const attackedImageFile = new File([blob], "attacked_image.jpg", { type: blob.type });
                formData.append('file', attackedImageFile);
                
                // Submit to extraction endpoint
                return fetch('/extract', {
                    method: 'POST',
                    body: formData
                });
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    extractionResult.innerHTML = `
                        <h4><i class="fas fa-exclamation-triangle"></i> Extraction Failed</h4>
                        <p>${data.message}</p>
                    `;
                    return;
                }
                
                // Show successful extraction
                extractionResult.innerHTML = `<h4><i class="fas fa-check-circle"></i> Watermark Extraction Results</h4>`;
                const contentDiv = document.createElement('div');
                contentDiv.className = 'extraction-content';
                
                if (data.has_watermark_image && data.watermark_image_url) {
                    // Create image element
                    const img = document.createElement('img');
                    img.src = data.watermark_image_url;
                    img.alt = "Extracted Watermark Image";
                    img.className = "extracted-watermark-image";
                    
                    // Create download link
                    const downloadLink = document.createElement('div');
                    downloadLink.className = 'watermark-download';
                    downloadLink.innerHTML = `<a href="${data.watermark_image_url}" class="btn btn-sm btn-secondary">
                        <i class="fas fa-download"></i> Download Extracted Watermark
                    </a>`;
                    
                    // Add elements to container
                    contentDiv.appendChild(img);
                    contentDiv.appendChild(downloadLink);
                    
                    // Add success message
                    const successMsg = document.createElement('div');
                    successMsg.className = 'success-message mt-3';
                    successMsg.innerHTML = `<i class="fas fa-shield-alt"></i> Your watermark survived the attack!`;
                    contentDiv.appendChild(successMsg);
                } else {
                    // No watermark found or text watermark
                    const message = document.createElement('div');
                    message.className = "text-content";
                    message.textContent = data.watermark || "No image watermark was found";
                    contentDiv.appendChild(message);
                }
                
                extractionResult.appendChild(contentDiv);
            })
            .catch(error => {
                extractionResult.innerHTML = `
                    <h4><i class="fas fa-exclamation-triangle"></i> Extraction Error</h4>
                    <p>Failed to extract watermark: ${error.message}</p>
                `;
            });
    }

    // Helper function to get file extension from MIME type
    function getFileExtension(mimeType) {
        switch(mimeType) {
            case 'image/jpeg':
                return '.jpg';
            case 'image/png':
                return '.png';
            case 'image/bmp':
                return '.bmp';
            case 'image/gif':
                return '.gif';
            default:
                return '.jpg';
        }
    }

    // Function to display detect error
    function showDetectError(message) {
        detectErrorText.textContent = message;
        detectProgressSection.classList.add('hidden');
        detectErrorSection.classList.remove('hidden');
        detectForm.classList.add('hidden');
    }

    // Handle "Detect Another File" button
    detectNewBtn.addEventListener('click', function() {
        resetDetectForm();
    });

    // Handle "Try Again" button for detect tab
    detectTryAgainBtn.addEventListener('click', function() {
        resetDetectForm();
    });

    // Function to reset the create form
    function resetCreateForm() {
        document.querySelector('#watermarkForm').classList.remove('hidden');
        createProgressSection.classList.add('hidden');
        createResultSection.classList.add('hidden');
        createErrorSection.classList.add('hidden');
        
        // Reset form
        watermarkForm.reset();
        fileName.textContent = 'No file selected';
        watermarkImageName.textContent = 'No image selected';
        
        // Reset media previews
        document.querySelector('.media-preview').innerHTML = '<div class="placeholder">Preview will appear here</div>';
        
        // Reset watermark preview
        const previewContainer = document.querySelector('.watermark-preview');
        if (previewContainer) {
            previewContainer.innerHTML = '<div class="placeholder">Preview will appear here</div>';
        }
        
        // Remove any controls if they exist
        const mediaControls = document.querySelector('.media-preview-container .preview-controls');
        if (mediaControls) mediaControls.remove();
        
        const resultControls = document.querySelector('.result-preview-container .preview-controls');
        if (resultControls) resultControls.remove();
    }

    // Function to reset the extract form
    function resetExtractForm() {
        extractForm.classList.remove('hidden');
        extractProgressSection.classList.add('hidden');
        extractResultSection.classList.add('hidden');
        extractErrorSection.classList.add('hidden');
        
        // Reset form
        extractForm.reset();
        extractFileName.textContent = 'No file selected';
        
        // Reset preview
        document.getElementById('extract-media-preview').innerHTML = '<div class="placeholder">Preview will appear here</div>';
        
        // Remove any controls if they exist
        const extractControls = document.querySelector('.extract-media-preview-container .preview-controls');
        if (extractControls) extractControls.remove();
    }

    // Function to reset the detect form
    function resetDetectForm() {
        detectForm.classList.remove('hidden');
        detectProgressSection.classList.add('hidden');
        detectResultSection.classList.add('hidden');
        detectErrorSection.classList.add('hidden');
        
        // Reset form
        detectForm.reset();
        detectFileName.textContent = 'No file selected';
        
        // Reset preview
        document.getElementById('detect-media-preview').innerHTML = '<div class="placeholder">Preview will appear here</div>';
        
        // Remove any controls if they exist
        const detectControls = document.querySelector('#detect-media-preview-container .preview-controls');
        if (detectControls) detectControls.remove();
        
        // Also hide attack extraction result
        const extractionResult = document.getElementById('attack-extraction-result');
        if (extractionResult) {
            extractionResult.classList.add('hidden');
        }
    }

    // Modified resetApp function to work with the new navigation
    function resetApp() {
        // Show create tab
        document.getElementById('create-tab').classList.add('active');
        document.getElementById('extract-tab').classList.remove('active');
        document.getElementById('detect-tab').classList.remove('active');
        
        // Update navigation - fix this to use side-nav instead of main-nav
        navItems.forEach(nav => nav.classList.remove('active'));
        document.querySelector('.side-nav [data-tab="create"]').classList.add('active');
        
        // Reset all forms
        resetCreateForm();
        resetExtractForm();
        resetDetectForm();
    }

    // Handle "Process Another File" button for create tab
    newWatermarkBtn.addEventListener('click', function() {
        resetCreateForm();
    });

    // Handle "Try Again" button for create tab
    createTryAgainBtn.addEventListener('click', function() {
        resetCreateForm();
    });

    // Handle "Extract Another File" button
    extractNewBtn.addEventListener('click', function() {
        resetExtractForm();
    });

    // Handle "Try Again" button for extract tab
    extractTryAgainBtn.addEventListener('click', function() {
        resetExtractForm();
    });

    // Handle position selector changes
    document.getElementById('position').addEventListener('change', function() {
        // Get the advanced positioning controls
        const advancedControls = document.getElementById('advanced-positioning');
        
        // Show or hide advanced controls based on selection
        if (this.value === 'custom') {
            advancedControls.style.display = 'block';
        } else {
            // If not custom, still keep the controls visible but update the sliders
            // to match the selected position
            advancedControls.style.display = 'block';
            
            // Update sliders based on selected position
            const xPosSlider = document.getElementById('x_pos');
            const yPosSlider = document.getElementById('y_pos');
            
            if (this.value === 'top-left') {
                xPosSlider.value = 10;
                yPosSlider.value = 10;
            } else if (this.value === 'top-right') {
                xPosSlider.value = 90;
                yPosSlider.value = 10;
            } else if (this.value === 'bottom-left') {
                xPosSlider.value = 10;
                yPosSlider.value = 90;
            } else if (this.value === 'bottom-right') {
                xPosSlider.value = 90;
                yPosSlider.value = 90;
            } else if (this.value === 'center') {
                xPosSlider.value = 50;
                yPosSlider.value = 50;
            }
            
            // Update displays
            document.getElementById('x-pos-value').textContent = xPosSlider.value + '%';
            document.getElementById('y-pos-value').textContent = yPosSlider.value + '%';
        }
    });

    // Add event listeners for advanced positioning sliders
    document.getElementById('x_pos').addEventListener('input', function() {
        document.getElementById('x-pos-value').textContent = this.value + '%';
        // Set position to custom if user is adjusting sliders
        document.getElementById('position').value = 'custom';
    });

    document.getElementById('y_pos').addEventListener('input', function() {
        document.getElementById('y-pos-value').textContent = this.value + '%';
        // Set position to custom if user is adjusting sliders
        document.getElementById('position').value = 'custom';
    });

    document.getElementById('rotation').addEventListener('input', function() {
        document.getElementById('rotation-value').textContent = this.value + '°';
    });

    document.getElementById('opacity').addEventListener('input', function() {
        document.getElementById('opacity-value').textContent = this.value + '%';
    });

    // Add event listener for attack type selection to show/hide relevant parameters
    document.getElementById('attack_type').addEventListener('change', function() {
        // Hide all parameter sections
        const paramSections = document.querySelectorAll('.attack-params');
        paramSections.forEach(section => section.classList.add('hidden'));
        
        // Show the relevant section for the selected attack
        const attackType = this.value;
        const relevantParamSection = document.getElementById(`attack-${attackType}-params`);
        
        // Handle combined attacks which don't have parameters
        if (attackType.startsWith('combined_') || attackType === 'equalization') {
            // No parameters to show
        } else if (relevantParamSection) {
            relevantParamSection.classList.remove('hidden');
        } else {
            // Default to compression params if not found
            const compressionParams = document.getElementById('attack-compression-params');
            if (compressionParams) {
                compressionParams.classList.remove('hidden');
            }
        }
    });

    // Ensure default attack parameters are shown on page load
    document.addEventListener('DOMContentLoaded', function() {
        // Show the default attack parameters when the detect tab is shown
        document.querySelector('[data-tab="detect"]').addEventListener('click', function() {
            // Get the currently selected attack type
            const attackType = document.getElementById('attack_type').value;
            
            // Hide all parameter sections first
            const paramSections = document.querySelectorAll('.attack-params');
            paramSections.forEach(section => section.classList.add('hidden'));
            
            // Show the correct parameter section
            const relevantParamSection = document.getElementById(`attack-${attackType}-params`);
            if (relevantParamSection) {
                relevantParamSection.classList.remove('hidden');
            }
        });
        
        // Manually trigger the change event for attack_type to set initial visibility
        const attackTypeSelect = document.getElementById('attack_type');
        if (attackTypeSelect) {
            const event = new Event('change');
            attackTypeSelect.dispatchEvent(event);
        }
    });

    // Update slider value displays
    document.getElementById('crop_percentage').addEventListener('input', function() {
        document.getElementById('crop-value').textContent = this.value + '%';
    });

    document.getElementById('noise_level').addEventListener('input', function() {
        document.getElementById('noise-value').textContent = this.value;
    });

    document.getElementById('blur_level').addEventListener('input', function() {
        document.getElementById('blur-value').textContent = this.value + 'x' + this.value;
    });

    document.getElementById('angle').addEventListener('input', function() {
        document.getElementById('angle-value').textContent = this.value + '°';
    });

    document.getElementById('brightness_factor').addEventListener('input', function() {
        document.getElementById('brightness-value').textContent = this.value + 'x';
    });

    document.getElementById('contrast_factor').addEventListener('input', function() {
        document.getElementById('contrast-value').textContent = this.value + 'x';
    });

    document.getElementById('kernel_size').addEventListener('input', function() {
        document.getElementById('kernel-value').textContent = this.value + 'x' + this.value;
    });

    document.getElementById('scale').addEventListener('input', function() {
        document.getElementById('scale-value').textContent = (this.value * 100).toFixed(0) + '%';
    });
});

// Update all download links to handle download without specifying filename
document.addEventListener('DOMContentLoaded', function() {
    // Find all download links and remove any download attributes
    document.querySelectorAll('a[href^="/download/"]').forEach(link => {
        link.removeAttribute('download');
    });
    
    document.querySelectorAll('a[href^="/download-temp/"]').forEach(link => {
        link.removeAttribute('download');
    });
    
    document.querySelectorAll('a[href^="/download-single/"]').forEach(link => {
        link.removeAttribute('download');
    });
});
