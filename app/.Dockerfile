FROM python:3.11-slim

WORKDIR /root/code

# Install dependencies first so Docker can cache this layer across rebuilds
COPY ./code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (main.py, pages/, models/, utils.py, tests)
COPY ./code /root/code

EXPOSE 8050

# Runs the Dash dev server directly. HOST/PORT are read from the environment
# (see docker-compose.yaml) 
CMD ["python3", "main.py"]
