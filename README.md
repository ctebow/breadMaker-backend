# Bread Maker Backend

This project is the backend for a web application that allows users to upload an image of a hand-drawn circuit diagram, and in return, receive a digital representation of the circuit, including component identification, connections, and a formatted layout. The backend uses a YOLOv8 model for component detection and OpenCV for line detection to determine the connections between components.

## Features

*   **Component Recognition:** Utilizes a YOLOv8 object detection model to identify various circuit components from an image.
*   **Connection Detection:** Employs OpenCV's Line Segment Detector (LSD) to identify wires and establish connections between components.
*   **Circuit Logic:** Processes the recognized components and connections to build a logical representation of the circuit.
*   **Formatted Layout:** Generates a formatted layout of the circuit components and wires, ready for visualization on a frontend.
*   **FastAPI Backend:** Provides a robust and fast backend server with clear API endpoints.

## How It Works

1.  **Image Upload:** The user uploads an image of a circuit diagram to the frontend, which then sends it to the backend.
2.  **YOLOv8 Detection:** The backend receives the image and uses a pre-trained YOLOv8 model to detect and classify circuit components (e.g., resistors, capacitors, voltage sources).
3.  **Line Detection:** OpenCV's Line Segment Detector (LSD) is used to detect lines in the image, which represent the wires connecting the components.
4.  **Post-processing:** The detected lines are processed to merge broken segments, remove noise, and trim lines that overlap with component bounding boxes.
5.  **Connection Mapping:** The processed lines are used to determine which components are connected to each other.
6.  **Layout Formatting:** The components and their connections are formatted into a structured JSON response, including positions, rotations, and snap points for easy rendering on a grid-based frontend.
7.  **API Response:** The backend sends the formatted JSON data back to the frontend for visualization.

## API Endpoints

### `/process`

*   **Method:** `POST`
*   **Description:** Receives a circuit image and a JSON object of component data, processes the image to detect components and connections, and returns a formatted JSON response.
*   **Request:**
    *   `file`: An image file of the circuit diagram.
    *   `json_data`: A JSON string containing initial component data.
*   **Response:**
    ```json
    {
      "status": "success",
      "components": { ... },
      "connections": { ... },
      "lines": [ ... ],
      "filename": "...",
      "time": "..."
    }
    ```

### `/update-components`

*   **Method:** `POST`
*   **Description:** Receives updated component data from the frontend (e.g., new resistor values) and processes it.
*   **Request:**
    ```json
    {
      "components": {
        "resistor_1": { "value": 1000 },
        ...
      }
    }
    ```
*   **Response:**
    ```json
    {
      "status": "success",
      "message": "Data received"
    }
    ```

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd breadMakerBackend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the backend server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## Model Information

*   **Model:** YOLOv8
*   **Training:** The model has been trained on a custom dataset of hand-drawn circuit diagrams.
*   **Performance:**
    *   **mAP@50:** 0.79
    *   **Box Loss:** 1.01
    *   **Class Loss:** 0.49
*   **Improvements:** The model is continuously being improved with more training data and better bounding box annotations.