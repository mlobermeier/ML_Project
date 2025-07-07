echo "Checking if Python is installed..."
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python to run this script."
    exit 1
fi

echo "Checking if required Python packages are installed..."
pip install -r requirements.txt

echo "Running full AirBnB price prediction pipeline..."
python full_pipeline.py
echo "Pipeline execution completed."

