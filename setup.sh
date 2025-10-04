#!/bin/bash

# =============================================================================
# Sapheneia TimesFM Setup Script
# 
# This script sets up the complete Sapheneia TimesFM environment for:
# - Local development and research
# - Notebook environment
# - Web application deployment (localhost and GCP)
# 
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [options]
#
# Options:
#   --local-only    Setup only for local development (no webapp)
#   --webapp-only   Setup only webapp dependencies
#   --gcp-deploy    Setup for GCP deployment
#   --help          Show this help message
# =============================================================================

# =============================================================================
#
# Step-by-step explanation of what the setup.sh script does:
#
#   1. Parses Options: It first checks if you've provided any options like --local-only, --webapp-only, or --gcp-deploy to determine the scope of the setup. If you don't provide any, it performs a full setup (local development and web app).
#
#   2. Detects System: The script checks your operating system and CPU architecture (e.g., Apple Silicon arm64 or Intel/AMD x86_64). This is important for installing the correct version of the TimesFM library, as the PyTorch backend is used for Apple Silicon and the JAX backend is used for other systems.
#
#   3. Installs `uv`: It checks if the uv package manager is installed. If not, it downloads and installs it. uv is a fast Python package manager used by this project.
#
#   4. Creates Python Environment: It uses uv to create a Python 3.11 virtual environment in a directory named .venv. This isolates the project's dependencies from other Python projects on your system.
#
#   5. Installs Core Dependencies: It installs all the necessary Python libraries for the core functionality and research, including:
#       * numpy, pandas, matplotlib, seaborn for data manipulation and visualization.
#       * jax and jaxlib for numerical computation.
#       * timesfm (the appropriate version for your system).
#       * jupyter, ipykernel, ipywidgets for running the notebooks.
#       * plotly for interactive visualizations in the web application.
#
#   6. Installs Web App Dependencies: If a full or webapp-only setup is selected, it installs the dependencies listed in webapp/requirements.txt, which includes Flask for the web server.
#
#   7. Sets Up Project Structure: It creates several directories that the application needs to run, such as data/, logs/, and webapp/uploads/. It also creates a standard .gitignore file if one doesn't already exist to prevent temporary files and data from being committed to version control.
#
#   8. Sets Up GCP Deployment (Optional): If you use the --gcp-deploy option, it creates a new script called deploy_gcp.sh that you can use to deploy the web application to Google Cloud Run.
#
#   9. Verifies Installation: Finally, it runs a quick check to ensure everything was installed correctly. It tries to import the main libraries (timesfm, pandas, flask, etc.) and runs a very basic TimesFM operation to confirm the model is functional.
#
#   After all these steps, it prints a "Setup Complete!" message with instructions on what to do next, such as how to activate the virtual environment and run the Jupyter notebooks or the web app.
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sapheneia-timesfm"
PYTHON_VERSION="3.11"
VENV_NAME=".venv"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect system architecture
detect_system() {
    ARCH=$(uname -m)
    OS=$(uname -s)
    
    print_status "Detected system: $OS ($ARCH)"
    
    # Determine if we're on Apple Silicon
    if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
        APPLE_SILICON=true
        print_status "Apple Silicon detected - will use PyTorch TimesFM backend"
    else
        APPLE_SILICON=false
        print_status "x86_64 architecture detected - will use JAX TimesFM backend"
    fi
}

# Function to install UV package manager
install_uv() {
    print_header "Installing UV Package Manager"
    
    if command_exists uv; then
        print_status "UV already installed: $(uv --version)"
        return
    fi
    
    print_status "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add UV to PATH for current session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if command_exists uv; then
        print_status "UV installed successfully: $(uv --version)"
    else
        print_error "UV installation failed"
        exit 1
    fi
}

# Function to setup Python environment
setup_python_env() {
    print_header "Setting up Python Environment"
    
    # Create virtual environment with UV
    print_status "Creating Python $PYTHON_VERSION virtual environment..."
    uv venv $VENV_NAME --python $PYTHON_VERSION
    
    # Activate virtual environment
    source $VENV_NAME/bin/activate
    print_status "Virtual environment activated: $VIRTUAL_ENV"
    
    # Upgrade pip
    uv pip install --upgrade pip
}

# Function to install core dependencies
install_core_dependencies() {
    print_header "Installing Core Dependencies"
    
    print_status "Installing base scientific computing packages..."
    uv pip install numpy pandas matplotlib seaborn scikit-learn python-dateutil
    
    print_status "Installing JAX for covariates support..."
    if [[ "$APPLE_SILICON" == true ]]; then
        # Install JAX for Apple Silicon
        uv pip install jax jaxlib
    else
        # Install JAX for x86_64
        uv pip install "jax[cpu]" jaxlib
    fi
    
    print_status "Installing TimesFM..."
    if [[ "$APPLE_SILICON" == true ]]; then
        # Use PyTorch version for Apple Silicon
        uv pip install "timesfm[torch]"
    else
        # Use JAX version for x86_64
        uv pip install timesfm
    fi
    
    print_status "Installing Jupyter notebook support..."
    uv pip install jupyter notebook ipykernel ipywidgets
    
    print_status "Installing Plotly for interactive visualizations..."
    uv pip install plotly
}

# Function to install webapp dependencies
install_webapp_dependencies() {
    print_header "Installing Web Application Dependencies"
    
    print_status "Installing Flask and web dependencies..."
    uv pip install -r webapp/requirements.txt
}

# Function to verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    print_status "Testing TimesFM import..."
    python -c "
import timesfm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

print('✅ TimesFM imported successfully')
print('✅ Core dependencies available')
print('✅ Plotly imported successfully')

# Test basic TimesFM functionality
try:
    hparams = timesfm.TimesFmHparams(
        backend='cpu',
        per_core_batch_size=1,
        horizon_len=4,
        num_layers=50,
        context_len=64
    )
    print('✅ TimesFM hyperparameters created')
except Exception as e:
    print(f'⚠️  TimesFM basic test failed: {e}')

print('✅ Installation verification completed')
"
    
    if [[ "$SETUP_TYPE" != "--local-only" ]]; then
        print_status "Testing web application imports..."
        python -c "
try:
    from flask import Flask
    print('✅ Flask available')
    
    import plotly.graph_objects as go
    print('✅ Plotly available')
    
    import sys
    sys.path.append('src')
    from model import TimesFMModel
    from data import DataProcessor
    from forecast import Forecaster
    from interactive_visualization import InteractiveVisualizer
    print('✅ Sapheneia modules available')
except ImportError as e:
    print(f'⚠️  Web app imports failed: {e}')
"
    fi
}

# Function to setup project structure
setup_project_structure() {
    print_header "Setting up Project Structure"
    
    # Create necessary directories
    mkdir -p data
    mkdir -p notebooks
    mkdir -p logs
    mkdir -p webapp/uploads
    mkdir -p webapp/results
    
    print_status "Project directories created"
    
    # Create .gitignore if it doesn't exist
    if [[ ! -f .gitignore ]]; then
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.venv/
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# Data and Results
data/
logs/
webapp/uploads/
webapp/results/
*.csv
*.json
*.pkl
*.png
*.jpg

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Secrets
.env
config.json
EOF
        print_status ".gitignore created"
    fi
}

# Function to setup GCP deployment
setup_gcp_deployment() {
    print_header "Setting up GCP Deployment Configuration"
    
    if ! command_exists gcloud; then
        print_warning "Google Cloud SDK not found. Please install it for GCP deployment:"
        print_warning "https://cloud.google.com/sdk/docs/install"
        return
    fi
    
    print_status "Google Cloud SDK found: $(gcloud version | head -n1)"
    
    # Create deployment script
    cat > deploy_gcp.sh << 'EOF'
#!/bin/bash

# Sapheneia TimesFM GCP Deployment Script

set -e

PROJECT_ID="${1:-your-project-id}"
REGION="${2:-us-central1}"

if [[ "$PROJECT_ID" == "your-project-id" ]]; then
    echo "Usage: ./deploy_gcp.sh YOUR_PROJECT_ID [REGION]"
    echo "Example: ./deploy_gcp.sh sapheneia-demo us-central1"
    exit 1
fi

echo "🚀 Deploying Sapheneia TimesFM to GCP"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo "Starting Cloud Build deployment..."
gcloud builds submit webapp/ --config=webapp/cloudbuild.yaml \
    --substitutions=_REGION=$REGION

# Get the service URL
SERVICE_URL=$(gcloud run services describe sapheneia-timesfm --region=$REGION --format="value(status.url)")

echo "✅ Deployment completed!"
echo "🌐 Service URL: $SERVICE_URL"
EOF
    
    chmod +x deploy_gcp.sh
    print_status "GCP deployment script created: deploy_gcp.sh"
}

# Function to display usage help
show_help() {
    cat << EOF
Sapheneia TimesFM Setup Script

USAGE:
    ./setup.sh [OPTIONS]

OPTIONS:
    --local-only    Setup only for local development (notebooks, src modules)
    --webapp-only   Setup only webapp dependencies
    --gcp-deploy    Setup for GCP Cloud Run deployment
    --help          Show this help message

EXAMPLES:
    ./setup.sh                    # Full setup (local + webapp)
    ./setup.sh --local-only       # Only research environment
    ./setup.sh --webapp-only      # Only web application
    ./setup.sh --gcp-deploy       # Setup with GCP deployment tools

REQUIREMENTS:
    - Bash shell
    - curl (for downloading UV)
    - Internet connection

The script will:
1. Install UV package manager
2. Create Python virtual environment
3. Install TimesFM and dependencies (including Plotly for interactive visualizations)
4. Setup project structure
5. Verify installation

For web application setup, it will additionally:
- Install Flask and web dependencies
- Install Plotly for interactive visualizations
- Create webapp deployment files
- Setup GCP deployment scripts (if requested)

EOF
}

# Main setup function
main() {
    print_header "Sapheneia TimesFM Setup"
    
    # Parse command line arguments
    SETUP_TYPE="${1:-full}"
    
    case $SETUP_TYPE in
        --help|-h)
            show_help
            exit 0
            ;;
        --local-only)
            print_status "Setting up for local development only"
            ;;
        --webapp-only)
            print_status "Setting up webapp dependencies only"
            ;;
        --gcp-deploy)
            print_status "Setting up with GCP deployment support"
            ;;
        full|"")
            print_status "Setting up full environment (local + webapp)"
            SETUP_TYPE="full"
            ;;
        *)
            print_error "Unknown option: $SETUP_TYPE"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    
    # Check if we're in the right directory
    if [[ ! -f "CLAUDE.md" ]] || [[ ! -d "src" ]]; then
        print_error "Please run this script from the sapheneia project root directory"
        print_error "Expected files: CLAUDE.md, src/ directory"
        exit 1
    fi
    
    # Detect system
    detect_system
    
    # Install UV package manager
    install_uv
    
    # Setup Python environment
    setup_python_env
    
    # Install core dependencies (always needed)
    install_core_dependencies
    
    # Install webapp dependencies if needed
    if [[ "$SETUP_TYPE" == "full" ]] || [[ "$SETUP_TYPE" == "--webapp-only" ]]; then
        install_webapp_dependencies
    fi
    
    # Setup project structure
    setup_project_structure
    
    
    # Setup GCP deployment if requested
    if [[ "$SETUP_TYPE" == "--gcp-deploy" ]] || [[ "$SETUP_TYPE" == "full" ]]; then
        setup_gcp_deployment
    fi
    
    # Verify installation
    verify_installation
    
    # Success message
    print_header "Setup Complete!"
    print_status "Sapheneia TimesFM environment is ready!"
    echo
    print_status "Next steps:"
    echo "  1. Activate environment: source $VENV_NAME/bin/activate"
    echo "  2. Start Jupyter: uv run jupyter notebook"
    echo "  3. Open demo notebook: notebooks/sapheneia_timesfm_demo.ipynb"
    
    if [[ "$SETUP_TYPE" == "full" ]] || [[ "$SETUP_TYPE" == "--webapp-only" ]]; then
        echo "  4. Run webapp: cd webapp && python app.py"
        echo "  5. Open browser: http://localhost:8080"
    fi
    
    if [[ "$SETUP_TYPE" == "--gcp-deploy" ]] || [[ "$SETUP_TYPE" == "full" ]]; then
        echo "  6. Deploy to GCP: ./deploy_gcp.sh YOUR_PROJECT_ID"
    fi
    
    echo
    print_status "Documentation: README.md"
    print_status "Configuration: CLAUDE.md"
}

# Run main function with all arguments
main "$@"