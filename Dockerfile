# Use a lightweight, official Python 3.12 image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy only the dependency files first to leverage Docker layer caching
COPY requirements.* ./

# Install dependencies
# We use an OR operator just in case you haven't run `uv pip compile` to generate requirements.txt yet
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir -r requirements.in

# Copy the rest of the application code (respecting .dockerignore)
COPY . /app

# Expose port 5000 for the Flask web server
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Pull DVC artifacts (if remote is configured) then start the app.
# The '|| true' ensures the app starts even if dvc pull fails.
CMD ["sh", "-c", "dvc pull || true && python app.py"]
