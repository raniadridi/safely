🔍 Safely - Fraud Detection System

A comprehensive machine learning-powered fraud detection system that analyzes financial transactions to identify potentially fraudulent activities in real-time.

## 🌟 Live Demo

**🚀 [Try Safely Live](https://safely-dim7.onrender.com)**

## 📊 Features

- **🤖 Advanced ML Models**: Multiple algorithms tested with GridSearch optimization
- **📈 99%+ Accuracy**: Achieved through balanced dataset and feature engineering
- **🎯 Real-time Predictions**: Instant fraud detection for transactions
- **📊 Smart Analytics**: Comprehensive data visualization and insights
- **💻 User-friendly Interface**: Clean, responsive web interface
- **⚡ Auto-calculation**: Smart form that calculates balance changes
- **🔒 Secure**: Built with security best practices

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 99.4% |
| **Precision** | 98.8% |
| **Recall** | 99.1% |
| **F1-Score** | 98.9% |

## 🏗️ Architecture

```
 Safely/
├── app.py                 
├── predictFraud.pkl                    
├──  static/
│   ├── css/main.css          
│   ├── img/                  
│   └── vendor/               
├──  templates/
│   ├── home.html             
│   ├── form.html             
│   ├── result.html           
│   └── visualization.html    
├──  fraudlent-transaction-prediction.ipynb
└──  requirements.txt       
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/safely.git
   cd safely
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

## 🔬 Model Training

The system uses an ensemble approach with multiple algorithms:

- **🌳 Random Forest** - Primary model
- **🚀 Gradient Boosting** - High accuracy alternative  
- **🌲 Decision Trees** - Interpretable baseline
- **📈 Logistic Regression** - Linear baseline
- **🎯 SVM** - Non-linear patterns
- **🔬 Extra Trees** - Additional ensemble

### Training Features

- **⚖️ Smart Undersampling**: Balanced dataset for better fraud detection
- **🔍 GridSearch Optimization**: Automated hyperparameter tuning
- **📊 Cross-validation**: Robust model validation
- **📈 Feature Engineering**: Optimized input features


### Model Configuration

The system automatically detects and loads:
- `predictFraud.pkl` - Main ML model
- `model_info.json` - Model metadata

## 📈 Performance Metrics

### Dataset Statistics
- **📊 Training Samples**: 120,000+
- **⚖️ Balanced Ratio**: 3:1 (No Fraud:Fraud)
- **🎯 Feature Count**: 4 optimized features
- **🔍 Cross-validation**: 5-fold CV

### Transaction Types Supported
- **💸 CASH_OUT**: Cash withdrawals
- **💳 PAYMENT**: Goods/services payments  
- **💰 CASH_IN**: Cash deposits
- **🔄 TRANSFER**: Account transfers
- **💳 DEBIT**: Direct debits

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **ML/AI**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Bootstrap 5, Custom CSS
- **Deployment**: Render, Gunicorn
- **Version Control**: Git, GitHub

## 📚 Model Development

### Jupyter Notebooks (Kaggle)

The model was developed and trained using Kaggle notebooks:

1. **📊 Data Analysis**: Comprehensive EDA with visualizations
2. **🔬 Feature Engineering**: Smart feature selection and scaling
3. **🤖 Model Training**: Multiple algorithms with GridSearch
4. **📈 Evaluation**: Performance metrics and validation
5. **💾 Model Export**: Saved for production deployment


## 🙏 Acknowledgments

- **Dataset**: Kaggle Fraudulent Transactions Dataset
- **Inspiration**: Real-world financial fraud detection needs
- **Libraries**: scikit-learn, Flask, Bootstrap communities

## 📞 Contact

- **👨‍💻 Developer**: Rania Dridi
- **📧 Email**: raniadridi42@gmail.com
- **🔗 LinkedIn**: [Rania Dridi](https://linkedin.com/in/raniadridii)
- **🐙 GitHub**: [Rania Dridi](https://github.com/raniadridi)

---

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for safer financial transactions