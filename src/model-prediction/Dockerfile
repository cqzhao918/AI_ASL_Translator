# Use the official Debian-hosted Python image
FROM python:3.9-slim-buster

ARG DEBIAN_PACKAGES="build-essential git curl wget unzip gzip"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

#ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/data-pipeline.json

# Ensure we have an up to date baseline, install dependencies 
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get install -y --no-install-recommends software-properties-common apt-transport-https ca-certificates gnupg2 gnupg-agent curl openssh-client && \
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-cloud-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app && \
    chown app:app /app

# Switch to the new user
USER app
WORKDIR /app

# # # Set the working directory to /preprocessing
# RUN pipenv lock

# # Add the Pipfile, Pipfile.lock, and python code into the container
# ADD . /

# RUN pipenv sync

# Install python packages
ADD --chown=app:app Pipfile Pipfile.lock /app/

RUN pipenv sync

# Make the entrypoint.sh script executable
# RUN chmod +x /bin/bash/entrypoint.sh
# Add the rest of the source code. This is done last so we don't invalidate all
# layers when we change a line of code.
ADD --chown=app:app . /app

# Make the entrypoint.sh script executable
# RUN chmod +x /bin/bash/entrypoint.sh

# # Set the entrypoint
# ENTRYPOINT ["/bin/bash"]

# # Specify the entrypoint script as the CMD
# CMD ["entrypoint.sh"]
# CMD ["-c", "pipenv shell"]
ENTRYPOINT ["/bin/bash","./entrypoint.sh"]
