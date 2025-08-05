# AI American Sign Language Translator

A real-time American Sign Language (ASL) translation system that uses computer vision and machine learning to interpret hand gestures and convert them to text.

## 🚀 Features

- **Real-time ASL Recognition**: Capture and interpret hand gestures in real-time
- **High Accuracy**: Trained on extensive ASL datasets for reliable translations
- **User-Friendly Interface**: Intuitive web application for easy interaction
- **API Integration**: RESTful API for seamless integration with other applications
- **Cloud Deployment**: Scalable deployment on Google Cloud Platform

## 🖥️ Application Interface

Our web application provides an intuitive interface for ASL translation:

![Frontend Interface](images/frontend-2.png)

*The main interface allows users to upload ASL videos and receive real-time translations.*

### Demo Video

Watch our application in action:

https://github.com/chuqi/AI_ASL_Translator/assets/chuqi/ac215-capy-demo.mp4

*This demo shows the complete workflow from video upload to ASL translation results.*

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Google Cloud Platform account (for deployment)
- Node.js 16+ (for frontend development)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI_ASL_Translator
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd src/frontend
   npm install
   ```

## 🏃‍♂️ Quick Start

### Running the API Service

1. Navigate to the API service directory:
   ```bash
   cd src/api-service
   ```

2. Start the API server:
   ```bash
   sh docker-shell.sh
   uvicorn_server
   ```

3. Access the API documentation at `http://localhost:9000/docs`

### Running the Frontend

1. Navigate to the frontend directory:
   ```bash
   cd src/frontend
   ```

2. Start the development server:
   ```bash
   sh docker-shell.sh
   yarn install  # First time only
   yarn start
   ```

3. Access the application at `http://localhost:3000`

## 🏗️ Project Structure

```
AI_ASL_Translator/
├── data/                          # Data storage (not uploaded to repo)
│   ├── interim/                   # Intermediate preprocessed data
│   ├── processed/                 # Final dataset files for modeling
│   └── raw/                      # Original immutable input data
├── notebooks/                     # Jupyter notebooks for EDA and testing
├── src/                          # Source code
│   ├── data-collector/           # Dataset creation scripts
│   ├── data-processor/           # Data processing code
│   ├── model-training/           # Model training and evaluation
│   ├── model-deploy/             # Model deployment
│   ├── workflow/                 # Automation scripts
│   ├── api-service/              # Backend API service
│   ├── frontend/                 # React frontend application
│   └── deployment/               # GCP deployment configuration
├── reports/                      # Generated reports and documentation
├── references/                   # Reference materials and papers
└── requirements.txt              # Python dependencies
```

## 🏛️ System Architecture

### Solution Architecture

Our solution architecture outlines the complete workflow from data collection to deployment:

![Solution Architecture](images/solution-arch.png)

*The solution architecture shows the three main layers: Process (People), Execution (Code), and State (Source, Data, Models).*

### Technical Architecture

The technical architecture details the implementation components and their interactions:

![Technical Architecture](images/technical-arch.png)

*The technical architecture illustrates the frontend, backend, ML pipeline, and state management components.*

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=9000
DEBUG=False

# Model Configuration
MODEL_PATH=./models/asl_model.pkl
CONFIDENCE_THRESHOLD=0.8

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/asl_db

# GCP Configuration (for deployment)
GCP_PROJECT_ID=your-project-id
GCP_ZONE=us-central1-a
```

## 🚀 Deployment

### Google Cloud Platform Deployment

The application can be deployed to GCP using Ansible playbooks:

1. **Build and push Docker images to GCR:**
   ```bash
   cd src/deployment
   ansible-playbook deploy-docker-images.yml -i inventory.yml
   ```

2. **Create GCP compute instance:**
   ```bash
   ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
   ```

3. **Provision the instance:**
   ```bash
   ansible-playbook deploy-provision-instance.yml -i inventory.yml
   ```

4. **Setup Docker containers:**
   ```bash
   ansible-playbook deploy-setup-containers.yml -i inventory.yml
   ```

5. **Configure webserver:**
   ```bash
   ansible-playbook deploy-setup-webserver.yml -i inventory.yml
   ```

Access your deployed application at `http://<EXTERNAL_IP>/`

![Deployment Architecture](images/deployment-single-vm.png)

*The application is deployed on a single VM in Google Cloud Platform with containerized services.*

## 📊 API Documentation

The API provides the following endpoints:

- `POST /predict` - Upload an image for ASL translation
- `GET /health` - Health check endpoint
- `GET /model-info` - Get model information and statistics
- `GET /docs` - Interactive API documentation (Swagger UI)

![API Endpoints](images/api-list.png)

*Available API endpoints for the ASL translation service.*

### Example API Usage

```python
import requests

# Upload image for translation
with open('asl_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:9000/predict', files=files)
    result = response.json()
    print(f"Translation: {result['translation']}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ASL dataset contributors
- Open source computer vision libraries
- Google Cloud Platform for hosting infrastructure

## 👨‍💻 Author

**Chuqing Zhao** - *AI American Sign Language Translator*

This project was developed as part of an AI/ML initiative to create accessible technology for the deaf and hard-of-hearing community.

## 📞 Support

For support and questions, please open an issue in the GitHub repository or contact the development team.

---

**Note**: This application is designed for educational and research purposes. For production use in critical applications, additional testing and validation is recommended. 
