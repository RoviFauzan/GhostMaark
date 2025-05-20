document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('watermarkForm');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress');
    const resultSection = document.getElementById('result');
    const downloadLink = document.getElementById('download-link');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show progress container
        progressContainer.classList.remove('hidden');
        resultSection.classList.add('hidden');
        
        // Get form data
        const formData = new FormData(form);
        
        // Submit form data
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                progressContainer.classList.add('hidden');
                return;
            }
            
            // Start checking the progress
            const taskId = data.task_id;
            checkProgress(taskId);
        })
        .catch(error => {
            alert('Error: ' + error.message);
            progressContainer.classList.add('hidden');
        });
    });
    
    function checkProgress(taskId) {
        fetch('/progress/' + taskId)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'error') {
                    alert('Error: ' + data.message);
                    progressContainer.classList.add('hidden');
                    return;
                }
                
                // Update progress bar
                progressBar.style.width = data.progress + '%';
                
                if (data.status === 'complete') {
                    // Show download button
                    progressContainer.classList.add('hidden');
                    resultSection.classList.remove('hidden');
                    downloadLink.href = '/download/' + taskId;
                    downloadLink.download = data.filename;
                } else {
                    // Check again after a short delay
                    setTimeout(() => checkProgress(taskId), 500);
                }
            })
            .catch(error => {
                alert('Error checking progress: ' + error.message);
                progressContainer.classList.add('hidden');
            });
    }
});
