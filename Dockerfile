# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies list and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all source code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]