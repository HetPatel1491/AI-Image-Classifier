# Use Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]