FROM python:3.9-slim-bullseye

WORKDIR /app

# Install dependencies needed for SQLite compilation and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libreadline-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    libffi-dev \
    zlib1g-dev

# Download and install the latest SQLite (required for ChromaDB)
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz \
    && tar -xzf sqlite-autoconf-3420000.tar.gz \
    && cd sqlite-autoconf-3420000 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3420000 \
    && rm sqlite-autoconf-3420000.tar.gz

# Set environment variable to use the newly installed SQLite
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy requirements or pyproject.toml first for better layer caching
COPY pyproject.toml ./
# Install dependencies - optimize for build time
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Fix numpy binary compatibility issue by installing specific versions
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir Cython==0.29.36

# Install Spacy after numpy is correctly installed
RUN pip install --no-cache-dir spacy==3.5.4
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
# Install German spaCy model
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.5.0/de_core_news_sm-3.5.0-py3-none-any.whl

# Now install the rest of the dependencies
RUN pip install --no-cache-dir .[all]

# Copy phentrieve package and API code
COPY phentrieve/ /app/phentrieve/
COPY api/ /app/api/

# Set environment variables
ENV PYTHONPATH=/app
# Container path where host data will be mounted
ENV PHENTRIEVE_DATA_ROOT_DIR=/phentrieve_data_mount
# These are derived from PHENTRIEVE_DATA_ROOT_DIR by the code
# Just setting them here for clarity about expected structure
ENV PHENTRIEVE_DATA_DIR=/phentrieve_data_mount/hpo_core_data
ENV PHENTRIEVE_INDEX_DIR=/phentrieve_data_mount/indexes
ENV PHENTRIEVE_RESULTS_DIR=/phentrieve_data_mount/results

# Clean up unnecessary packages after installation to reduce image size
RUN apt-get purge -y --auto-remove build-essential wget libreadline-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev \
    libbz2-dev libffi-dev zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
