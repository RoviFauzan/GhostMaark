from flask import render_template, send_file, jsonify, request
import os
import uuid

def setup_main_routes(app, config, tasks):
    """Mengatur rute utama aplikasi"""
    
    @app.route('/')
    def index():
        """Halaman utama"""
        return render_template('index.html')
    
    @app.route('/progress/<task_id>')
    def get_progress(task_id):
        """Mendapatkan progres dari task"""
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify(tasks[task_id])
    
    @app.route('/download/<task_id>')
    def download_file(task_id):
        """Download file hasil watermark"""
        if task_id not in tasks or tasks[task_id]['status'] != 'complete':
            return jsonify({'error': 'File not ready or task not found'}), 404
        
        output_path = tasks[task_id]['output_path']
        return send_file(output_path, as_attachment=True)
    
    @app.route('/preview/<task_id>')
    def preview_file(task_id):
        """Preview file hasil watermark"""
        if task_id not in tasks or tasks[task_id]['status'] != 'complete':
            return jsonify({'error': 'File not ready or task not found'}), 404
        
        output_path = tasks[task_id]['output_path']
        filename = os.path.basename(output_path)
        
        # Import di sini untuk menghindari circular import
        from utils.file_utils import get_mime_type
        mime_type = get_mime_type(output_path)
        
        # Buat URI untuk preview
        preview_url = f"/temp-preview/{task_id}/{filename}"
        
        return jsonify({
            'preview_url': preview_url,
            'mime_type': mime_type
        })
    
    @app.route('/temp-preview/<task_id>/<filename>')
    def serve_preview(task_id, filename):
        """Menyajikan file sementara untuk preview"""
        if task_id not in tasks or tasks[task_id]['status'] != 'complete':
            return jsonify({'error': 'File not ready or task not found'}), 404
        
        output_path = tasks[task_id]['output_path']
        
        # Cek apakah filename yang diminta sesuai dengan output filename sebenarnya
        if os.path.basename(output_path) != filename:
            return jsonify({'error': 'Invalid filename'}), 404
        
        # Untuk keamanan, verifikasi bahwa file berada di dalam OUTPUT_FOLDER
        if not output_path.startswith(config.OUTPUT_FOLDER):
            return jsonify({'error': 'Security error: Invalid file path'}), 403
        
        # Sajikan file (tanpa memicu download)
        return send_file(output_path)
    
    @app.route('/download-temp/<filename>')
    def download_temp_file(filename):
        """Download file sementara (hasil ekstraksi)"""
        file_path = os.path.join(config.OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Cek apakah file dalam OUTPUT_FOLDER
        if not file_path.startswith(config.OUTPUT_FOLDER):
            return jsonify({'error': 'Security error: Invalid file path'}), 403
        
        return send_file(file_path)
