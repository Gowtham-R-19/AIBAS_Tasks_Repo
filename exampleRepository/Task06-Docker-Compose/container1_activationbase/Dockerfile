# Use lightweight base image
FROM busybox:latest

# Create directory for activation data
RUN mkdir -p /activationBase

# Copy the single data entry
COPY currentActivation.csv /activationBase/currentActivation.csv

# Set working directory
WORKDIR /activationBase

# Default command to display the file
CMD ["cat", "/activationBase/currentActivation.csv"]
