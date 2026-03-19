print("HELLO — worker started")

import redis
import time
import json
import subprocess
import os

INPUT_DIR = os.path.abspath("data")  # your local data folder
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

QUEUE_NAME = "ocr_jobs"

print("Connected to Redis. Waiting for jobs...")

while True:
    job = r.blpop(QUEUE_NAME, timeout=5)

    if job:
        _, data = job
        try:
            job_data = json.loads(data)
        except:
            job_data = data

        print("\n--- New Job ---")
        print(job_data)

        input_file = job_data.get("file")
        output_file = input_file.replace(".pdf", "_ocr.pdf")

        input_path = os.path.join(INPUT_DIR, input_file)
        output_path = os.path.join(INPUT_DIR, output_file)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{INPUT_DIR}:/data",
            "ocr-base",
            "ocrmypdf",
            "-l", "eng",
            "--rotate-pages",
            "--deskew",
            "--optimize", "2",
            "--jobs", "4",
            f"/data/{input_file}",
            f"/data/{output_file}"
        ]

        print("Running OCR:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            print("OCR completed:", output_file)
        except subprocess.CalledProcessError as e:
            print("OCR failed:", e)
    else:
        print("No jobs... waiting")
        time.sleep(1)