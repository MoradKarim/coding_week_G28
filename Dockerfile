# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variable for Python to prevent output buffering
ENV PYTHONUNBUFFERED=1

# Command to run the application (start Streamlit)
CMD ["streamlit", "run", "app/app.py"]
