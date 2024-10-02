# Set the NVIDIA-PyTorch image as default
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
ADD . /workspace

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install PyTorch, torchvision, and other relevant packages
RUN pip install torch torchvision torchaudio

# Ensure scikit-learn is installed for metrics
RUN pip install scikit-learn

# Expose port 80 to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME DiabeticRetinopathyClassifier
