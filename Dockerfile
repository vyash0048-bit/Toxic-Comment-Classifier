# Use a lightweight, official Python 3.12 image
FROM python:3.12-slim

# Set up a new user named "user" with user ID 1000 (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy only the dependency files first to leverage Docker layer caching
COPY --chown=user:user requirements.* ./

# Switch to the "user" user
USER user

# Install dependencies in the user's local directory
RUN pip install --user --no-cache-dir -r requirements.txt || pip install --user --no-cache-dir -r requirements.in

# Copy the rest of the application code
COPY --chown=user:user . $HOME/app

# Expose port 7860 for the Flask web server
EXPOSE 7860

# Pull DVC artifacts (if remote is configured) then start the app.
# The '|| true' ensures the app starts even if dvc pull fails.
CMD ["sh", "-c", "dvc pull || true && python app.py"]
