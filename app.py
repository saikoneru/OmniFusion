from flask import Flask, request, jsonify
import os
import sys
import time
import queue
import threading
import traceback

# OmniFusion repo is cloned into /app at build time
sys.path.append("/app")

from omni_fusion_model import OmniFusionModel

# ---------------------------------------------------------------------------
# Config — driven by environment variables for Docker compatibility
# ---------------------------------------------------------------------------
CACHE_DIR      = os.getenv("HF_HOME", "/app/hf_cache")
MODEL_NAME     = os.getenv("MODEL_NAME", "skoneru/OmniFusion_v2")
HOST           = os.getenv("FLASK_HOST", "0.0.0.0")
PORT           = int(os.getenv("FLASK_PORT", "8088"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "2"))
BATCH_TIMEOUT  = float(os.getenv("BATCH_TIMEOUT", "0.05"))  # seconds

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print(f"Loading OmniFusionModel — checkpoint: {MODEL_NAME}, cache: {CACHE_DIR}")
system = OmniFusionModel(checkpoint_path=MODEL_NAME, cache_dir=CACHE_DIR)
print("Model loaded successfully.")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def process_input(audios, prefixes, langs, images):
    """
    Process a batch of requests.

    Each item is one of:
      - audio-only:    audio=<base64 wav>, image=None
      - audio + image: audio=<base64 wav>, image=<PIL Image>
      - text + image:  audio=None,         image=<PIL Image>, prefix=<source text>

    Returns a list of hypothesis strings, one per input.
    """
    hypos = []

    for audio, prefix, lang, image in zip(audios, prefixes, langs, images):
        # Pass base64 audio as a data URI; image is already a PIL Image or None.
        audio_input = [f"data:audio/wav;base64,{audio}"] if audio is not None else []
        image_input = [image] if image is not None else []
        # Per OmniFusion docs: source_text must be "" when audio is provided
        source_text = [prefix if not audio_input else ""]

        print(f"  lang={lang}, has_audio={bool(audio_input)}, has_image={bool(image_input)}")

        result = system.translate(
            audio_paths=audio_input,
            image_paths=image_input,
            source_texts=source_text,
            target_lang=lang,
            num_beams=3,
        )
        hypos.append(result[0] if result else "")

    return hypos

# ---------------------------------------------------------------------------
# Request batching
# ---------------------------------------------------------------------------
app = Flask(__name__)
request_queue = queue.Queue()
processing = True

def batch_worker():
    global processing

    while processing:
        batch = []

        # Wait for first request (blocking)
        try:
            first_item = request_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        batch.append(first_item)

        # Pre-batch window — collect more requests up to MAX_BATCH_SIZE
        pre_deadline = time.time() + BATCH_TIMEOUT
        while len(batch) < MAX_BATCH_SIZE and time.time() < pre_deadline:
            try:
                item = request_queue.get(timeout=max(0.001, pre_deadline - time.time()))
                batch.append(item)
            except queue.Empty:
                break

        def run_batch(batch_items):
            print(f"Processing batch of {len(batch_items)} request(s)")
            audios   = [it["audio"]       for it in batch_items]
            prefixes = [it["prefix"]      for it in batch_items]
            langs    = [it["output_lang"] for it in batch_items]
            images   = [it["image"]       for it in batch_items]
            try:
                hypos = process_input(audios, prefixes, langs, images)
                for item, hypo in zip(batch_items, hypos):
                    item["result_queue"].put({"success": True, "hypos": hypo})
            except Exception as e:
                error_msg = f"Batch processing error: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                for item in batch_items:
                    item["result_queue"].put({"success": False, "error": error_msg})

        run_batch(batch)

        # Post-batch window — pick up requests that arrived during inference
        post_batch = []
        post_deadline = time.time() + BATCH_TIMEOUT
        while len(post_batch) < MAX_BATCH_SIZE and time.time() < post_deadline:
            try:
                item = request_queue.get(timeout=max(0.001, post_deadline - time.time()))
                post_batch.append(item)
            except queue.Empty:
                break

        if post_batch:
            run_batch(post_batch)


worker_thread = threading.Thread(target=batch_worker, daemon=True)
worker_thread.start()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/v1/chat/completions", methods=["POST"])
def process_batch():
    data = request.get_json()

    if not data or ("audio" not in data and "audios" not in data):
        return jsonify({"error": "Missing field: audios/audio"}), 400

    if "audio" in data:
        # Single request — enqueue and wait
        result_queue = queue.Queue()
        request_item = {
            "audio":        data["audio"],
            "prefix":       data.get("prefix", ""),
            "output_lang":  data.get("output_langs", data.get("output_lang", "English")),
            "image":        data.get("image", None),
            "result_queue": result_queue,
        }
        request_queue.put(request_item)

        try:
            result = result_queue.get(timeout=30.0)
            if result["success"]:
                return jsonify({"hypos": result["hypos"]})
            else:
                with open("log.txt", "a") as f:
                    f.write(f"Exception: {result['error']}\n")
                return jsonify({"error": "Internal server error"}), 500
        except queue.Empty:
            return jsonify({"error": "Request timeout"}), 500

    else:
        # Batch request — process directly (bypasses queue)
        audios       = data["audios"]
        prefixes     = data.get("prefixes", [""] * len(audios))
        output_langs = data.get("output_langs", ["English"] * len(audios))
        images       = data.get("images", [None] * len(audios))

        try:
            hypos = process_input(audios, prefixes, output_langs, images)
            return jsonify({"hypos": hypos})
        except Exception as e:
            with open("log.txt", "a") as f:
                f.write(f"Exception: {str(e)}\n")
                f.write(traceback.format_exc())
            return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting server on {HOST}:{PORT} (max_batch={MAX_BATCH_SIZE}, timeout={BATCH_TIMEOUT}s)")
    try:
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    finally:
        processing = False
        worker_thread.join(timeout=2.0)
