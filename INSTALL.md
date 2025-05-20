# Watermarker Installation Instructions

## Basic Installation

1. Clone the repository or download the files
2. Make sure you have Python 3.7+ installed
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

You can run the application in two ways:

### Option 1: Using the helper script (recommended)

```bash
python run.py
```

This script will:
- Check for missing dependencies
- Offer to install them for you
- Start the application with appropriate settings

### Option 2: Running directly

```bash
python Watermarker.py
```

## Optional Dependencies

- **TensorFlow**: Required for deep learning watermarking features.
  If you don't need these features, you can skip installing TensorFlow.

## Troubleshooting

If you encounter the error `ModuleNotFoundError: No module named 'tensorflow'`, you have two options:

1. Install TensorFlow to enable all features:
   ```bash
   pip install tensorflow
   ```

2. Continue without TensorFlow - the application will run with reduced functionality.
   The deep learning watermarking options will be disabled.

## System Requirements

- Python 3.7 or higher
- At least 2GB of RAM (4GB+ recommended if using TensorFlow)
- Disk space: 200MB (2GB+ if installing TensorFlow)
