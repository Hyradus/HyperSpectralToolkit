# Use the official miniconda image from Docker Hub
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app


# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libopengl0 \
    chromium-driver\
    chromium

# Copy the environment.yml file into the Docker image
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Activate the environment and ensure it's available in the PATH
RUN echo "conda activate crism_toolkit" >> ~/.bashrc
ENV PATH /opt/conda/envs/crism_toolkit/bin:$PATH

# Copy your Bokeh application code into the Docker image
COPY ./app /app

# Expose the port that Bokeh server will run on
EXPOSE 5006
WORKDIR /app
# Define the command to run your Bokeh application
#CMD ["bokeh", "serve", "crism_toolkit.py", "--allow-websocket-origin=*", "--port=5006"]
CMD ["bokeh", "serve", "crism_toolkit.py", "--allow-websocket-origin=*", "--port=5006"]

