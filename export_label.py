import pytesseract
import cv2
import json
from pathlib import Path


def extract_text_and_boxes(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert image to grayscale for better OCR accuracy
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR with Tesseract
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    # Process the extracted data
    results = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:  # Ignore empty strings
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            results.append({
                "text": text,
                "bbox": [x, y, x + w, y + h]
            })
    
    return results

def save_label_studio_format(image_path, output_json):
    image_name = Path(image_path).name
    extracted_data = extract_text_and_boxes(image_path)
    
    label_studio_data = {
        "data": {
            "image": image_name,
            "annotations": [{
                "result": [
                    {
                        "value": {
                            "text": [entry["text"]],
                            "coordinates": entry["bbox"]
                        },
                        "from_name": "transcription",
                        "to_name": "image",
                        "type": "textarea"
                    } for entry in extracted_data
                ]
            }]
        }
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(label_studio_data, f, indent=4)

if __name__ == "__main__":
    image_path = "/Users/tesla/side/lg_data/OCR/test_png.png"  # Change this to your image path
    output_json = "/Users/tesla/side/lg_data/OCR/label_studio_output.json"
    save_label_studio_format(image_path, output_json)
    print(f"Data saved in {output_json}")

