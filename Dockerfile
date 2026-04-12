FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir gunicorn

COPY requirements/dev.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY SharedBackend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

COPY --from=build /src /src
COPY app app
COPY SharedBackend SharedBackend
COPY assets assets
COPY alembic.ini alembic.ini
COPY migrations migrations
COPY entrypoint.sh entrypoint.sh
ENV PYTHONPATH=/src/app/:/src/SharedBackend/src/

RUN chmod +x entrypoint.sh

EXPOSE 8080
CMD ["./entrypoint.sh"]
