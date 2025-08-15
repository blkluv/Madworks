FROM python:3.11-slim

# Install system dependencies for CairoSVG (cairo/pango/gdk-pixbuf) and fonts
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libcairo2 \
       libpango-1.0-0 \
       libpangocairo-1.0-0 \
       libgdk-pixbuf-2.0-0 \
       libffi8 \
       libjpeg62-turbo \
       fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]


