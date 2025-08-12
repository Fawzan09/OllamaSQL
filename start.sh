#!/bin/bash

# OllamaSQL Flask Application Startup Script

echo "🦙 Starting OllamaSQL Flask Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if Ollama is running
echo "Checking if Ollama server is running..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama server is running"
else
    echo "⚠️  Ollama server is not running. Please start Ollama with: ollama serve"
    echo "   The application will still work for file upload and management, but AI features will be disabled."
fi

# Start Flask application
echo "🚀 Starting Flask application at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
python app.py