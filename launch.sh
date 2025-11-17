#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    if [ ! -z "$API_PID" ]; then
        echo -e "\n${YELLOW}Shutting down FastAPI server (PID: $API_PID)...${NC}"
        kill $API_PID 2>/dev/null
        wait $API_PID 2>/dev/null
    fi
    echo -e "${GREEN}✅ Cleanup complete${NC}"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

echo "🚀 RAG Pipeline Launcher"
echo "========================"

# Step 1: Check if Python exists
echo -e "\n${YELLOW}Step 1: Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found. Please install Python first.${NC}"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"

# Step 2: Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found!${NC}"
    exit 1
fi

# Step 3: Check if pip exists
echo -e "\n${YELLOW}Step 2: Checking pip installation...${NC}"
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo -e "${RED}❌ pip not found. Installing pip...${NC}"
    $PYTHON_CMD -m ensurepip --upgrade
fi

# Step 4: Install requirements
echo -e "\n${YELLOW}Step 3: Installing/updating dependencies from requirements.txt...${NC}"
$PYTHON_CMD -m pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install requirements${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 5: Check if key libraries are installed
echo -e "\n${YELLOW}Step 4: Verifying key libraries...${NC}"
$PYTHON_CMD -c "import streamlit, fastapi, uvicorn, langchain" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Key libraries not found. Please check your installation.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Key libraries found${NC}"

# Step 6: Find API file
echo -e "\n${YELLOW}Step 5: Locating API file...${NC}"
if [ -f "api.py" ]; then
    API_FILE="api.py"
elif [ -f "experimental_files/api.py" ]; then
    API_FILE="experimental_files/api.py"
else
    echo -e "${RED}❌ api.py not found in root or experimental_files directory${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Found API file: $API_FILE${NC}"

# Step 7: Start FastAPI server in background
echo -e "\n${YELLOW}Step 6: Starting FastAPI server...${NC}"
API_PID=""
if [ "$API_FILE" == "api.py" ]; then
    uvicorn api:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
    API_PID=$!
else
    # If api.py is in experimental_files, add to PYTHONPATH and run from there
    export PYTHONPATH="${PWD}/experimental_files:${PYTHONPATH}"
    cd experimental_files
    uvicorn api:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
    API_PID=$!
    cd - > /dev/null
fi
echo -e "${GREEN}✅ FastAPI server started (PID: $API_PID)${NC}"

# Step 8: Check if curl is available for health check
if ! command -v curl &> /dev/null; then
    echo -e "${YELLOW}⚠️  curl not found. Skipping health check.${NC}"
    echo -e "${YELLOW}   Please verify the API is running manually at http://localhost:8000${NC}"
    HEALTH_CHECK_PASSED=true
else
    # Wait for server to be ready and check health
    echo -e "\n${YELLOW}Step 7: Checking API health status...${NC}"
    MAX_RETRIES=30
    RETRY_COUNT=0
    HEALTH_CHECK_PASSED=false

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        sleep 1
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null || echo "000")
        
        if [ "$HTTP_CODE" == "200" ]; then
            HEALTH_CHECK_PASSED=true
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -n "."
    done

    echo ""

    if [ "$HEALTH_CHECK_PASSED" = true ]; then
        echo -e "${GREEN}✅ API health check passed!${NC}"
        echo -e "${GREEN}   API is running at http://localhost:8000${NC}"
    else
        echo -e "${RED}❌ API health check failed after $MAX_RETRIES seconds${NC}"
        echo -e "${YELLOW}   Attempting to kill background process...${NC}"
        kill $API_PID 2>/dev/null
        exit 1
    fi
fi

# Step 9: Launch Streamlit frontend
echo -e "\n${YELLOW}Step 8: Launching Streamlit frontend...${NC}"
echo -e "${GREEN}✅ Starting Streamlit app (app_api.py)${NC}"
echo -e "${GREEN}   Frontend will be available at http://localhost:8501${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop both servers${NC}\n"

# Launch Streamlit (this will block)
# When Streamlit exits, the trap will handle cleanup
streamlit run app_api.py

