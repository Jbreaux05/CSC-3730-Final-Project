# Artist Recognition Project

This is our final project for Machine Learning CSC 3730.
This repo consists of a machine learning backend via Flask, and a modern web frontend built with Next.js (React).

## Project Structure


The project is (somewhat) divided into two main components.
* **Root Directory (Backend):** Contains the Flask server (`index.py`) and the training/testing logic.
* **Frontend Directory:** Contains the Next.js React application.

## Prerequisites

* **Python 3.8+**
* **Node.js 18+** & **npm**

## Setup Instructions

To run the demo, you will need two separate terminal windows open.

### 1. Backend Setup (Terminal 1)

The Flask server (`index.py`) acts as the API. It must be running for the frontend to connect.

1. Open your terminal in the **root** directory of the project.
2. Set up a virtual environment (optional, but extremely recommended):
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\Activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Start the Server:
    ```bash
    python index.py
    ```
    *Keep this terminal window open.*

### 2. Frontend Step (Terminal 2)

The frontend is a Next.js application located in the `Frontend/` folder.

1. Open a new terminal window.
2. Navigate to the frontend directory:
    ```bash
    cd Frontend
    ```
3. Install the Node modules:
    ```bash
    npm install
    ```
4. Start the Web Application:
    ```bash
    npm run dev
    ```

## Usage

Once both terminals ar e running:

1. Open your web browser.
2. Navigate to `http://localhost:3000` (or the port shown in your Terminal 2).
3. This website will automatically connect to the Flask server running in the background.
4. Play the game.