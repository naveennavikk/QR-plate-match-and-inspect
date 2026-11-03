# QR Plate Matching + Inspector

This combined Streamlit application runs the reference plate matcher as soon as both images are uploaded. It asks for a reference plate image and one property image. The matcher combines multiscale visual matching with Grounding DINO QR-code localization, then compares the surrounding background at the located candidate. The background check includes color, texture direction, and surface detail when the candidate is at a similar scale. It does not read QR payloads or printed plate IDs. Blurry or too-small source images, or a low-confidence QR localization, show REQUIRED MANUAL REVIEW. Confident matches produce ACCEPT or REJECT based on the background check. Only ACCEPT runs the QR plate inspector on the reference plate image.

## Start

Install the combined dependencies in a Python environment:

```powershell
python -m pip install -r requirements.txt
```

Run the application:

```powershell
streamlit run app.py
```

The app folder contains the matcher code and inspector model weights it needs, so it can run independently of the original project folders.
