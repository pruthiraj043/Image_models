# Use an official Python image as the base
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (if applicable, change as needed)
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
