from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import email
from email.policy import default
from langdetect import detect, DetectorFactory
from typing import List, Dict, Any
import os
import logging

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Spam Email Detector",
    description="Advanced spam detection API with AI-powered features",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class EmailRequest(BaseModel):
    text: str
    sender: str

class SpamResponse(BaseModel):
    label: str
    confidence: float
    spammy_keywords: List[str]
    language: str
    sender_risk: str
    bot_likeness: str
    explanation: str

# Global variables for model and vectorizer
model = None
vectorizer = None

# [Keep all your existing constants: SPAM_KEYWORDS, BOT_PHRASES, HIGH_RISK_PATTERNS]
SPAM_KEYWORDS = [
    'win', 'winner', 'won', 'prize', 'lottery', 'congratulations',
    'free', 'click', 'urgent', 'act now', 'limited time', 'offer expires',
    'guaranteed', 'no risk', 'money back', 'earn money', 'work from home',
    'make money fast', 'get rich', 'financial freedom', 'investment opportunity',
    'viagra', 'cialis', 'pharmacy', 'prescription', 'medication',
    'weight loss', 'lose weight', 'diet pill', 'miracle cure',
    'dear customer', 'dear friend', 'dear sir/madam', 'valued customer',
    'you have been selected', 'claim now', 'claim your', 'verify account',
    'suspended account', 'update payment', 'confirm identity',
    'nigerian prince', 'inheritance', 'beneficiary', 'transfer funds',
    'casino', 'gambling', 'bet now', 'jackpot', 'slots',
    'adult', 'dating', 'singles', 'lonely', 'meet women',
    'refinance', 'mortgage', 'loan approved', 'credit score', 'debt relief',
    '100% free', '100% guaranteed', 'no strings attached', 'risk free',
    'call now', 'order now', 'buy now', 'subscribe now', 'register now'
]

BOT_PHRASES = [
    'dear customer', 'dear valued customer', 'dear sir/madam', 'dear friend',
    'act now', 'limited time', 'offer expires', 'don\'t miss out',
    'this is not spam', 'remove from list', 'unsubscribe',
    'you have been selected', 'congratulations you have won',
    'claim your prize', 'verify your account', 'update your information',
    'click here now', 'visit our website', 'call immediately',
    'order within', 'supplies are limited', 'while supplies last',
    'satisfaction guaranteed', 'money back guarantee', 'risk free trial'
]

HIGH_RISK_PATTERNS = [
    '.ru', '.tk', '.ml', '.ga', '.cf', '.biz', '.info',
    'tempmail', 'guerrillamail', '10minutemail', 'mailinator',
    'disposable', 'throwaway', 'temp-mail', 'fake-mail',
    'no-reply', 'noreply', 'donotreply'
]

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    try:
        # Try to load the model file
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, dict):
                    model = model_data.get('model')
                    vectorizer = model_data.get('vectorizer')
                else:
                    # Assume it's just the model
                    model = model_data
                    # Create a dummy vectorizer for demonstration
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found. Creating dummy model for demo purposes.")
            # Create dummy model for demonstration
            create_dummy_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        create_dummy_model()

def create_dummy_model():
    """Create a dummy model for demonstration when model.pkl is not available"""
    global model, vectorizer
    try:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create dummy components
        model = MultinomialNB()
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Dummy training data for the demo
        dummy_texts = [
            "Win a free iPhone now! Click here to claim your prize!",
            "Meeting scheduled for tomorrow at 3 PM in conference room A",
            "Congratulations! You've won $1,000,000 in our lottery!",
            "Please review the attached document and provide feedback",
            "URGENT: Verify your account immediately or it will be suspended!"
        ]
        dummy_labels = [1, 0, 1, 0, 1]  # 1 = spam, 0 = not spam
        
        # Fit the vectorizer and model
        X = vectorizer.fit_transform(dummy_texts)
        model.fit(X, dummy_labels)
        
        logger.info("Dummy model created for demonstration")
    except Exception as e:
        logger.error(f"Failed to create dummy model: {e}")
        raise

# [Keep all your existing helper functions: detect_spam_keywords, check_sender_risk, etc.]
def detect_spam_keywords(text: str) -> List[str]:
    """Detect spam keywords in the email text"""
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in SPAM_KEYWORDS:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords

def check_sender_risk(sender: str) -> str:
    """Check if sender email indicates high risk"""
    sender_lower = sender.lower()
    
    for pattern in HIGH_RISK_PATTERNS:
        if pattern in sender_lower:
            return "High Risk"
    
    return "Low Risk"

def detect_bot_likeness(text: str) -> str:
    """Detect if email appears to be bot-written"""
    text_lower = text.lower()
    bot_score = 0
    
    for phrase in BOT_PHRASES:
        if phrase in text_lower:
            bot_score += 1
    
    # Additional checks for bot-like patterns
    if re.search(r'dear (sir|madam|customer|friend)', text_lower):
        bot_score += 2
    
    if re.search(r'(click|call|order|buy|subscribe) (here|now|today)', text_lower):
        bot_score += 2
    
    if re.search(r'\d+% (free|off|discount|guaranteed)', text_lower):
        bot_score += 1
    
    return "Likely Bot-Written" if bot_score >= 3 else "Likely Human-Written"

def detect_language(text: str) -> str:
    """Detect the language of the email text"""
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

def generate_explanation(
    label: str, 
    confidence: float, 
    spammy_keywords: List[str], 
    sender_risk: str, 
    bot_likeness: str
) -> str:
    """Generate human-readable explanation for the prediction"""
    explanations = []
    
    if label == "SPAM":
        if confidence > 90:
            explanations.append("This email is very likely spam based on our AI analysis.")
        elif confidence > 70:
            explanations.append("This email appears to be spam with high confidence.")
        else:
            explanations.append("This email shows spam characteristics.")
        
        if spammy_keywords:
            keyword_list = ", ".join(spammy_keywords[:5])  # Show first 5 keywords
            explanations.append(f"Contains suspicious keywords: {keyword_list}.")
        
        if sender_risk == "High Risk":
            explanations.append("The sender's email domain appears suspicious.")
        
        if bot_likeness == "Likely Bot-Written":
            explanations.append("The writing style suggests automated/bot generation.")
    
    else:  # NOT SPAM
        if confidence > 90:
            explanations.append("This email appears to be legitimate with high confidence.")
        else:
            explanations.append("This email looks clean but has some minor indicators to monitor.")
        
        if spammy_keywords:
            explanations.append("Contains some flagged words but overall context appears legitimate.")
        
        if sender_risk == "High Risk":
            explanations.append("While the sender domain has some risk factors, the content appears legitimate.")
    
    if not explanations:
        explanations.append("Standard email analysis completed.")
    
    return " ".join(explanations)

def predict_spam(text: str, sender: str) -> Dict[str, Any]:
    """Main prediction function"""
    try:
        if not model or not vectorizer:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        # Vectorize the text
        text_vectorized = vectorizer.transform([text])
        
        # Get prediction and probability
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Determine label and confidence
        if prediction == 1:
            label = "SPAM"
            confidence = float(probabilities[1] * 100)
        else:
            label = "NOT SPAM"
            confidence = float(probabilities[0] * 100)
        
        # Run additional analysis
        spammy_keywords = detect_spam_keywords(text)
        sender_risk = check_sender_risk(sender)
        bot_likeness = detect_bot_likeness(text)
        language = detect_language(text)
        
        # Generate explanation
        explanation = generate_explanation(
            label, confidence, spammy_keywords, sender_risk, bot_likeness
        )
        
        return {
            "label": label,
            "confidence": round(confidence, 2),
            "spammy_keywords": spammy_keywords,
            "language": language,
            "sender_risk": sender_risk,
            "bot_likeness": bot_likeness,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def extract_email_content(eml_content: bytes) -> Dict[str, str]:
    """Extract text content and sender from .eml file"""
    try:
        # Parse the email
        msg = email.message_from_bytes(eml_content, policy=default)
        
        # Extract sender
        sender = msg.get('From', 'unknown@unknown.com')
        
        # Extract text content
        text_content = ""
        
        if msg.is_multipart():
            for part in msg.iter_parts():
                if part.get_content_type() == 'text/plain':
                    text_content += part.get_content()
                elif part.get_content_type() == 'text/html':
                    # Basic HTML stripping for demo purposes
                    html_content = part.get_content()
                    text_content += re.sub('<[^<]+?>', '', html_content)
        else:
            text_content = msg.get_content()
        
        return {
            "text": text_content,
            "sender": sender
        }
    
    except Exception as e:
        logger.error(f"Email parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse email file: {str(e)}")

# Fixed startup event (FastAPI 0.104.1 compatible)
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        # Try to create/train model if it doesn't exist
        if not os.path.exists('model.pkl'):
            logger.info("No model file found, attempting to create one...")
            try:
                from model import create_production_model
                create_production_model()
                load_model()  # Reload after creation
            except Exception as e:
                logger.warning(f"Could not create production model: {e}")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Spam Email Detector API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Analyze email content for spam",
            "/upload_eml": "POST - Upload .eml file for analysis",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/predict", response_model=SpamResponse)
async def predict_email_spam(request: EmailRequest):
    """
    Predict if an email is spam or not
    
    - **text**: Raw email content
    - **sender**: Sender email address
    """
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty")
    
    if not request.sender.strip():
        raise HTTPException(status_code=400, detail="Sender email cannot be empty")
    
    try:
        result = predict_spam(request.text, request.sender)
        return SpamResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/upload_eml")
async def upload_eml_file(file: UploadFile = File(...)):
    """
    Upload a .eml email file for spam analysis
    
    - **file**: .eml email file
    """
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith('.eml'):
        raise HTTPException(status_code=400, detail="File must be a .eml email file")
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract email content and sender
        extracted = extract_email_content(content)
        
        # Predict spam
        result = predict_spam(extracted["text"], extracted["sender"])
        
        # Add file info to response
        result["file_info"] = {
            "filename": file.filename,
            "size_bytes": len(content),
            "extracted_sender": extracted["sender"]
        }
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
