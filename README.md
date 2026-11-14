\# EcoSort AI



\*\*AI-Powered Waste Classification and Sorting Assistance\*\*



EcoSort AI is a dual-model waste-classification prototype designed to help users identify common waste materials and understand responsible disposal options.



\## How It Works



EcoSort AI combines two AI systems:



1\. \*\*MobileNetV2:\*\* A trained image-classification model that predicts the waste category.

2\. \*\*Google Gemini:\*\* Provides an independent AI review when the service is available.



The system compares their predictions and can flag uncertain results for review.



\## Supported Waste Categories



\- Cardboard

\- Glass

\- Metal

\- Paper

\- Plastic

\- Trash



\## Key Features



\- Image-based waste classification

\- Independent AI review

\- Prediction comparison and confidence display

\- Low-confidence and review-required alerts

\- Disposal guidance

\- Analytics dashboard

\- Recent scan activity



\## Technology Stack



\- Python

\- FastAPI

\- TensorFlow / Keras

\- MobileNetV2

\- Google Gemini API

\- HTML, CSS and JavaScript

\- SQLite



\## Running Locally



1\. Clone or download this repository.

2\. Install the required Python packages.

3\. Create a `.env` file and add your Gemini API key.

4\. Start the application:



```bash

python -m uvicorn VAI:app --reload

```



5\. Open `http://127.0.0.1:8000/` in your browser.



\*\*Important:\*\* Never publish your `.env` file or API key.



\## Important Notes



\- Predictions are AI-generated suggestions, not guaranteed classifications.

\- Model confidence is not the same as measured accuracy.

\- Gemini review depends on API availability and usage limits.

\- A scan does not confirm that an item was actually recycled.



\## Project Purpose



EcoSort AI explores how artificial intelligence can support waste identification, environmental awareness, and more informed disposal decisions.



\*\*Built for the Veritas AI International High School Hackathon 2026.\*\*

