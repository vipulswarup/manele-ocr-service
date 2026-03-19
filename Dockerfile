FROM ubuntu:25.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-ara \
    tesseract-ocr-pan \
    tesseract-ocr-urd \
    ocrmypdf \
    ghostscript \
    imagemagick \
    pngquant \
    unpaper \
    qpdf \
    poppler-utils \
    && apt-get clean
# Create working dir
WORKDIR /app

# Default command (we’ll override later)
CMD ["bash"]