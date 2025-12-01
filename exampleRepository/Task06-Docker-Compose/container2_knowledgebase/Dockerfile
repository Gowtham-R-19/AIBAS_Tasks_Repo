# Use lightweight base image
FROM busybox:latest

# Create directory for AI model
RUN mkdir -p /knowledgeBase

# Copy the AI model
COPY currentSolution.pkl /knowledgeBase/currentSolution.pkl

# Set working directory
WORKDIR /knowledgeBase

# Default command to list files
CMD ["ls", "-lh", "/knowledgeBase/"]