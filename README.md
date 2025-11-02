# 🏦 Loan Eligibility Predictor

An end-to-end machine learning project that predicts loan approval status using applicant information. This project includes data preprocessing, model training, a web interface, CI/CD pipeline, and Docker containerization.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## 🚀 Live Demo

[Try the live demo here](https://ml-loan.onrender.com/)

## 🌟 Features

- **Automated ML Pipeline**: Complete data preprocessing, feature engineering, and model training
- **Multiple Models**: Logistic Regression (baseline) and XGBoost (advanced)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics
- **REST API**: Flask-based API for predictions
- **Modern Web UI**: Clean, responsive HTML/CSS/JS interface
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Containerized**: Docker support for easy deployment
- **Production Ready**: Includes error handling, logging, and health checks

## 📊 Project Structure

```
ml-loan/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml      # CI/CD pipeline
├── src/
│   ├── preprocessing.py         # Data preprocessing module
│   └── train.py                 # Model training module
├── tests/
│   ├── test_preprocessing.py    # Preprocessing tests
│   ├── test_train.py           # Training tests
│   └── test_api.py             # API tests
├── templates/
│   └── index.html              # Web interface
├── static/
│   ├── style.css               # Styles
│   └── script.js               # Frontend logic
├── models/                      # Trained models (generated)
├── data/
│   └── processed/              # Processed data (generated)
├── reports/                     # Evaluation reports (generated)
├── app.py                      # Flask application
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Docker (optional)

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ml-loan
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run preprocessing**:
```bash
python src/preprocessing.py
```

4. **Train models**:
```bash
python src/train.py
```

5. **Start the web application**:
```bash
python app.py
```

6. **Open your browser**:
Navigate to `http://localhost:5000`

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t loan-predictor .
```

### Run Container

```bash
docker run -p 5000:5000 loan-predictor
```

### Push to Docker Hub

```bash
docker tag loan-predictor your-username/loan-predictor:latest
docker push your-username/loan-predictor:latest
```

## 📈 Model Performance

The project trains two models and automatically selects the best one:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.78 | ~0.81 | ~0.92 | ~0.86 | ~0.71 |
| XGBoost | ~0.79 | ~0.82 | ~0.92 | ~0.87 | ~0.73 |

*Note: Actual metrics may vary based on the dataset and random seed.*

## 🔬 Features Used

### Input Features
- **Gender**: Male/Female
- **Married**: Yes/No
- **Dependents**: 0, 1, 2, 3+
- **Education**: Graduate/Not Graduate
- **Self Employed**: Yes/No
- **Applicant Income**: Numeric
- **Coapplicant Income**: Numeric
- **Loan Amount**: Numeric (in thousands)
- **Loan Amount Term**: Numeric (in months)
- **Credit History**: 1 (Good) / 0 (Bad)
- **Property Area**: Urban/Semiurban/Rural

### Engineered Features
- Total Income
- Income to Loan Ratio
- Loan Amount per Term
- Log-transformed Income and Loan Amount

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:

1. ✅ Installs dependencies
2. ✅ Runs preprocessing and training
3. ✅ Executes unit tests
4. ✅ Generates coverage reports
5. ✅ Builds Docker image
6. ✅ Pushes to Docker Hub
7. ✅ Triggers deployment

### Setting Up CI/CD

Add these secrets to your GitHub repository:

- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password/token
- `RENDER_API_KEY`: Render API key (for deployment)
- `RENDER_SERVICE_ID`: Render service ID

## 🌐 API Endpoints

### Health Check
```http
GET /health
```

### Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "Name": "John Doe",
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
```

**Response**:
```json
{
  "success": true,
  "prediction": "Approved",
  "confidence": 0.89,
  "probability_approved": 0.89,
  "probability_rejected": 0.11,
  "applicant_name": "John Doe"
}
```

### Get Model Info
```http
GET /model-info
```

## 🚀 Deployment Options

### 1. Render
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Render will automatically detect the Dockerfile
4. Deploy!

### 2. Railway
1. Create a new project on Railway
2. Connect your GitHub repository
3. Add environment variables if needed
4. Deploy!

### 3. Heroku
```bash
heroku create your-app-name
heroku container:push web
heroku container:release web
```

### 4. AWS/GCP/Azure
Use the Docker image with your preferred cloud platform's container service.

## 📝 Dataset

The project uses the Loan Prediction Dataset which includes:
- 614 loan applications
- 13 features (including target variable)
- Binary classification (Approved/Rejected)

**Target Variable**: `Loan_Status` (Y = Approved, N = Rejected)

## 🛠️ Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Flask**: Web framework
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **pytest**: Testing framework
- **Docker**: Containerization
- **GitHub Actions**: CI/CD
- **HTML/CSS/JS**: Frontend

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Your Name - [Your GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset from Kaggle Loan Prediction Dataset
- Inspired by real-world loan approval systems
- Built with best practices for ML deployment

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!
