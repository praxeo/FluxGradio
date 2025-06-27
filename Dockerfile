# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Gradio will run on
EXPOSE 7860

# Set the command to run your application
CMD ["python", "app.py"]