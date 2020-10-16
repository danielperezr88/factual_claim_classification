FROM python:3.9

RUN apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install --no-install-recommends apt-utils && \
    apt-get -y install nano make sudo && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_ws.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements_ws.txt && \
    rm -rf /tmp/pip-tmp

# Commented out so that remoteUser is sudo user
# USER vscode
