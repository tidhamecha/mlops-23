 # Use a base image with the necessary environment (e.g., Python)
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app


# Copy the required files for model training to the working directory
COPY . /app

# Install necessary libraries
RUN pip install -r requirements.txt

# Create a volume to save the trained models on the host machine
VOLUME /app/models

# Start the model training when the container starts
CMD ["python", "train_models.py"]
