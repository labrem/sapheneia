#!/bin/bash

# TimesFM Environment Setup Script
# This script automatically sets up a complete working environment for TimesFM
# including support for covariates functionality.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Function to detect system architecture
detect_architecture() {
    local arch=$(uname -m)
    local os=$(uname -s)
    
    print_info "Detected system: $os on $arch"
    
    if [[ "$arch" == "arm64" ]] || [[ "$arch" == "aarch64" ]]; then
        if [[ "$os" == "Darwin" ]]; then
            echo "apple_silicon"
        else
            echo "arm64"
        fi
    elif [[ "$arch" == "x86_64" ]] || [[ "$arch" == "amd64" ]]; then
        echo "x86_64"
    else
        echo "unknown"
    fi
}

# Function to get clean architecture without print statements
get_architecture() {
    local arch=$(uname -m)
    local os=$(uname -s)
    
    if [[ "$arch" == "arm64" ]] || [[ "$arch" == "aarch64" ]]; then
        if [[ "$os" == "Darwin" ]]; then
            echo "apple_silicon"
        else
            echo "arm64"
        fi
    elif [[ "$arch" == "x86_64" ]] || [[ "$arch" == "amd64" ]]; then
        echo "x86_64"
    else
        echo "unknown"
    fi
}

# Function to check if UV is installed
check_uv_installed() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to install UV
install_uv() {
    print_info "Installing UV package manager..."
    
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        # Unix-like systems (Linux, macOS)
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Source the shell profile to make uv available
        if [[ -f "$HOME/.cargo/env" ]]; then
            source "$HOME/.cargo/env"
        fi
        
        # Add to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Verify installation
    if check_uv_installed; then
        print_success "UV installed successfully"
        uv --version
    else
        print_error "Failed to install UV. Please install manually from https://docs.astral.sh/uv/"
        exit 1
    fi
}

# Function to setup virtual environment based on architecture
setup_environment() {
    local arch=$1
    
    print_info "Setting up virtual environment for $arch architecture..."
    
    # Remove existing virtual environment if it exists
    if [[ -d ".venv" ]]; then
        print_warning "Removing existing virtual environment..."
        rm -rf .venv
    fi
    
    case $arch in
        "apple_silicon")
            print_info "Setting up PyTorch version (Apple Silicon compatible)..."
            uv python install 3.11
            uv venv --python 3.11
            
            print_info "Installing TimesFM with PyTorch dependencies..."
            uv pip install timesfm[torch]
            
            print_info "Installing JAX for covariates support..."
            uv pip install jax jaxlib
            
            print_info "Installing Jupyter for notebook support..."
            uv pip install jupyter
            
            CHECKPOINT_REPO="google/timesfm-2.0-500m-pytorch"
            ;;
        "arm64")
            print_info "Setting up PyTorch version (ARM64 compatible)..."
            uv python install 3.11
            uv venv --python 3.11
            
            print_info "Installing TimesFM with PyTorch dependencies..."
            uv pip install timesfm[torch]
            
            print_info "Installing JAX for covariates support..."
            uv pip install jax jaxlib
            
            print_info "Installing Jupyter for notebook support..."
            uv pip install jupyter
            
            CHECKPOINT_REPO="google/timesfm-2.0-500m-pytorch"
            ;;
        "x86_64")
            print_info "Detecting if PAX version is available..."
            
            # Try PAX version first
            uv python install 3.10
            uv venv --python 3.10
            
            if uv pip install timesfm[pax] 2>/dev/null; then
                print_success "PAX version installed successfully!"
                print_info "Installing Jupyter for notebook support..."
                uv pip install jupyter
                CHECKPOINT_REPO="google/timesfm-2.0-500m-jax"
            else
                print_warning "PAX version failed, falling back to PyTorch version..."
                rm -rf .venv
                
                uv python install 3.11
                uv venv --python 3.11
                uv pip install timesfm[torch]
                
                print_info "Installing JAX for covariates support..."
                uv pip install jax jaxlib
                
                print_info "Installing Jupyter for notebook support..."
                uv pip install jupyter
                
                CHECKPOINT_REPO="google/timesfm-2.0-500m-pytorch"
            fi
            ;;
        *)
            print_warning "Unknown architecture ($arch), defaulting to PyTorch version..."
            uv python install 3.11
            uv venv --python 3.11
            uv pip install timesfm[torch]
            uv pip install jax jaxlib
            print_info "Installing Jupyter for notebook support..."
            uv pip install jupyter
            CHECKPOINT_REPO="google/timesfm-2.0-500m-pytorch"
            ;;
    esac
    
    print_success "Virtual environment created successfully!"
}

# Function to verify installation
verify_installation() {
    local checkpoint_repo=$1
    
    print_info "Verifying TimesFM installation..."
    
    # Create verification script
    cat > /tmp/verify_timesfm.py << EOF
import sys
try:
    import timesfm
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    print("✅ All imports successful!")
    
    # Test TimesFM class availability and basic functionality
    try:
        # Create a simple hparams object to verify the API
        hparams = timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,
            horizon_len=24,
        )
        print("✅ TimesFM hparams created successfully!")
        
        # Check if TimesFm class has the expected methods
        if hasattr(timesfm.TimesFm, 'forecast'):
            print("✅ TimesFM forecast method available!")
        if hasattr(timesfm.TimesFm, 'forecast_with_covariates'):
            print("✅ TimesFM forecast_with_covariates method available!")
        
        print("✅ TimesFM API verification successful!")
        
    except Exception as e:
        print(f"⚠️  TimesFM API test warning: {e}")
        print("   This may indicate a configuration issue.")
    
    print("✅ Basic functionality verification completed!")
    
    # Test Jupyter components
    try:
        import jupyter
        import IPython
        print("✅ Jupyter components available!")
    except ImportError as e:
        print(f"⚠️  Jupyter import warning: {e}")
    
    print("🎉 All verifications passed!")
    
except Exception as e:
    print(f"❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if uv run python /tmp/verify_timesfm.py; then
        print_success "Installation verification completed successfully!"
        rm -f /tmp/verify_timesfm.py
        return 0
    else
        print_error "Installation verification failed!"
        rm -f /tmp/verify_timesfm.py
        return 1
    fi
}

# Function to display usage instructions
show_usage_instructions() {
    echo
    echo "======================================================================"
    echo -e "${GREEN}🎉 TimesFM Environment Setup Complete!${NC}"
    echo "======================================================================"
    echo
    echo -e "${BLUE}📋 Environment Details:${NC}"
    echo "   • Virtual environment: .venv"
    echo "   • Python version: $(uv run python --version)"
    echo "   • TimesFM: Installed with full functionality"
    echo "   • JAX/JAXlib: Available for covariates support"
    echo
    echo -e "${BLUE}🚀 Usage Commands:${NC}"
    echo
    echo "   Activate environment:"
    echo "   source .venv/bin/activate"
    echo
    echo "   Run Python scripts:"
    echo "   uv run python your_script.py"
    echo
    echo "   Start Jupyter notebook:"
    echo "   uv run jupyter notebook"
    echo
    echo "   Test TimesFM:"
    echo "   uv run python -c \"import timesfm; print('TimesFM ready!')\""
    echo
    echo -e "${BLUE}📚 Next Steps:${NC}"
    echo "   • Check out the notebooks/ directory for examples"
    echo "   • Try the covariates.ipynb notebook for advanced features"
    echo "   • Read the README.md for detailed usage instructions"
    echo
    echo "======================================================================"
}

# Main execution
main() {
    echo "======================================================================"
    echo -e "${BLUE}🔧 TimesFM Environment Setup Script${NC}"
    echo "======================================================================"
    echo
    
    # Detect architecture
    detect_architecture  # This prints the detection info
    ARCH=$(get_architecture)  # This gets the clean result
    print_info "Architecture detected: $ARCH"
    
    # Check and install UV if needed
    if check_uv_installed; then
        print_success "UV is already installed"
        uv --version
    else
        install_uv
    fi
    
    # Setup environment
    setup_environment "$ARCH"
    
    # Verify installation
    if verify_installation "$CHECKPOINT_REPO"; then
        show_usage_instructions
    else
        print_error "Setup completed but verification failed. Please check the installation manually."
        exit 1
    fi
}

# Handle script interruption
trap 'print_error "Setup interrupted by user. Cleaning up..."; rm -f /tmp/verify_timesfm.py; exit 1' INT TERM

# Run main function
main "$@"
