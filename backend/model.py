import pickle
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import requests
import io
import zipfile
import os


def load_enron_dataset():
    """Load the Enron spam dataset if available"""
    try:
        
        spam_path = "./data/enron_spam/"
        ham_path = "./data/enron_ham/"
        
        emails = []
        labels = []
        
        # Load spam emails
        if os.path.exists(spam_path):
            for filename in os.listdir(spam_path)[:2000]:  # Limit for demo
                try:
                    with open(os.path.join(spam_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        emails.append(f.read())
                        labels.append(1)
                except:
                    continue
        
        # Load ham emails  
        if os.path.exists(ham_path):
            for filename in os.listdir(ham_path)[:1000]:  # Limit for demo
                try:
                    with open(os.path.join(ham_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        emails.append(f.read())
                        labels.append(0)
                except:
                    continue
        
        if emails:
            print(f"Loaded {len(emails)} emails from Enron dataset")
            return emails, labels
        else:
            print("Enron dataset not found, using enhanced sample data")
            return None, None
            
    except Exception as e:
        print(f"Error loading Enron dataset: {e}")
        return None, None

def create_enhanced_sample_data():
    """Create a much larger and more realistic sample dataset"""
    
    # More sophisticated spam examples
    spam_emails = [
        # Phishing attempts
        "Dear Customer, Your PayPal account has been limited. Click here to verify: http://fake-paypal.com/verify",
        "URGENT: Your bank account will be closed. Update your information immediately at: secure-bank-update.net",
        "Your Amazon order #12345 has been cancelled. Verify your payment method: amazon-verify.tk",
        "IRS Notice: You owe $5,000 in taxes. Pay immediately to avoid arrest: irs-payment.ru",
        "Your Apple ID has been compromised. Reset your password: apple-security.ml",
        
        # Nigerian prince / advance fee fraud
        "Greetings, I am Prince John from Nigeria. I need your help to transfer $10 million dollars.",
        "Dear Friend, I have $25 million USD to transfer. I need your bank account details for processing.",
        "CONFIDENTIAL BUSINESS PROPOSAL: Help me move inherited funds and earn 30% commission.",
        "You have been chosen to receive inheritance from deceased relative in Africa worth $8.5M",
        "Urgent Business Proposal: Oil contract in Nigeria needs foreign partner. $50M profit guaranteed.",
        
        # Lottery/Prize scams
        "CONGRATULATIONS! You've won £850,000 in the UK National Lottery! Claim code: UK2024WIN",
        "WINNER NOTIFICATION: You are the lucky winner of $1,000,000 Microsoft email lottery!",
        "You have won a brand new iPhone 15 Pro Max! Click to claim your prize within 24 hours!",
        "MEGA JACKPOT WINNER: Your email won $5,000,000 in our international sweepstakes!",
        "Facebook Lottery Winner: You won $750,000! Contact agent immediately with winning code FB2024",
        
        # Cryptocurrency scams
        "Elon Musk is giving away FREE Bitcoin! Send 0.1 BTC to get 2 BTC back! Limited time!",
        "URGENT: Your crypto wallet needs verification. Click here or lose access forever!",
        "Make $10,000 daily trading Bitcoin with our AI bot. Join thousands of successful traders!",
        "WARNING: Your Bitcoin account expires in 24 hours. Verify now to keep your coins!",
        "Exclusive: Join Elon's crypto giveaway event. Double your investment guaranteed!",
        
        # Romance scams
        "My dear, I am a lonely widow with $2M inheritance. I need someone to trust with my fortune.",
        "Hello beautiful, I am US soldier in Afghanistan. I have gold bars worth $5M to share with you.",
        "Dearest one, I saw your profile and fell in love. I want to send you $50,000 as gift.",
        "My beloved, I am dying of cancer and want to donate my $3M fortune to you for charity work.",
        
        # Tech support scams
        "VIRUS ALERT: Your computer is infected with 18 viruses! Call +1-800-SCAMMER immediately!",
        "Microsoft Security Alert: Your Windows license has expired. Renew now or lose access!",
        "Your computer has been hacked! Call our tech support to remove malware: 1-888-FAKE-TECH",
        "WARNING: Suspicious activity detected on your IP address. Download security software now!",
        
        # Investment scams
        "Make $500-$5000 daily working from home! No experience needed! Start earning today!",
        "EXCLUSIVE: Buy Amazon stock before it splits! Our insider tip can make you millions!",
        "Guaranteed 300% returns in 30 days! Join our elite investment club. Limited spots available!",
        "Secret trading algorithm beats the market! Turn $100 into $10,000 in one month!",
        
        # Product scams
        "MIRACLE CURE: Lose 50 pounds in 30 days with our secret weight loss formula!",
        "Doctors HATE this simple trick that cures diabetes permanently! Click to learn more!",
        "FREE TRIAL: Revolutionary anti-aging cream makes you look 20 years younger!",
        "BREAKING: Local mom discovers wrinkle-removing trick. Cosmetic companies furious!",
        
        # Job scams
        "Earn $3000 weekly working from home! No skills required! Apply now and start today!",
        "URGENT: Personal assistant needed. $500/day salary. Must be honest and trustworthy.",
        "Work from home opportunity: Process payments for international company. $2000/week guaranteed!",
        "Mystery shopper position available. Get paid to shop! $300 per assignment!",
        
        # Gift card scams
        "You've won a $500 Walmart gift card! Complete our survey to claim your prize!",
        "FREE $1000 Amazon gift card for the first 100 people! Click here to get yours!",
        "Congratulations! You qualify for a free $250 Target gift card. Claim within 24 hours!",
        "Special offer: Get a $100 iTunes gift card absolutely free! No purchase necessary!",
        
        # Fake charities
        "Help orphaned children in Africa. Your $50 donation provides food for a month. Donate now!",
        "URGENT: Earthquake victims need your help. 100% of donations go to relief efforts.",
        "Save the endangered tigers! Your donation is tax-deductible and saves lives!",
        "Hurricane relief fund: Donate now to help families rebuild their lives.",
        
        # Social media scams
        "Your Facebook account will be deleted! Verify your identity to keep your account active!",
        "Instagram is shutting down! Share this message to 20 friends to keep your account!",
        "WhatsApp will start charging users! Forward this message to avoid monthly fees!",
        "BREAKING: TikTok giving away $10,000 to random users! Check if you're selected!",
        
        # Travel scams
        "FLASH SALE: 7-day Caribbean cruise for only $99! Book now, limited availability!",
        "You've won a free vacation to Hawaii! All expenses paid! Claim your trip now!",
        "LAST MINUTE DEAL: Luxury resort stay for $1/night! This offer expires tonight!",
        "FREE airfare to anywhere in the world! Complete our survey to claim your tickets!",
    ]
    
    # Realistic legitimate emails
    ham_emails = [
        # Work/Business emails
        "Hi John, can you please send me the quarterly sales report by end of day Friday?",
        "Meeting reminder: Project kickoff meeting tomorrow at 2 PM in conference room A.",
        "Thank you for your presentation yesterday. The client was very impressed with our proposal.",
        "Please review the attached contract and let me know if you have any questions or concerns.",
        "The software deployment is scheduled for this weekend. Please be available for testing on Monday.",
        "Great job on the project delivery! The client feedback has been overwhelmingly positive.",
        "Reminder: All expense reports are due by the 15th of each month for timely reimbursement.",
        "The new employee onboarding session is scheduled for next Tuesday at 10 AM.",
        "Please confirm your attendance at the annual company retreat in Colorado next month.",
        "The server maintenance window has been moved to Saturday night to minimize business impact.",
        
        # Personal emails
        "Hey mom, I'll be home for Thanksgiving dinner. Can't wait to see everyone!",
        "Happy birthday Sarah! Hope you have a wonderful day and celebration tonight.",
        "Thanks for recommending that restaurant. We had a fantastic dinner there last weekend.",
        "Are we still on for the hiking trip this Saturday? Weather forecast looks perfect!",
        "Congratulations on your new job! Let's celebrate this weekend if you're free.",
        "The kids loved the birthday party yesterday. Thank you for organizing everything so well.",
        "Reminder: Book club meeting is next Thursday. We're discussing 'The Silent Patient'.",
        "I found your wallet in my car. Let me know when you want me to drop it off.",
        "The vacation photos from Italy are amazing! Thanks for sharing them with everyone.",
        "Can you pick up some groceries on your way home? I'll send you the shopping list.",
        
        # Educational/Professional
        "Your course registration for Advanced Python Programming has been confirmed for spring semester.",
        "The research paper you submitted has been accepted for publication in the journal.",
        "Workshop reminder: 'Introduction to Machine Learning' starts at 9 AM in room 205.",
        "Your student loan payment of $350 has been successfully processed for this month.",
        "Congratulations! You have been selected for the summer internship program at our company.",
        "The conference abstract deadline has been extended to March 15th due to popular demand.",
        "Your thesis defense is scheduled for April 20th at 2 PM. Please confirm your availability.",
        "Welcome to the advanced certification program. Your course materials are now available online.",
        "The scholarship application you submitted is currently under review by our committee.",
        "Reminder: Final exams are scheduled for the week of May 15-19. Good luck with your studies!",
        
        # Financial/Banking (legitimate)
        "Your monthly statement for checking account ending in 1234 is now available online.",
        "Automatic payment of $1,250 for your mortgage has been successfully processed today.",
        "Your credit card payment of $450 has been received and posted to your account.",
        "Annual fee waiver has been applied to your premium rewards credit card account.",
        "Your direct deposit of $3,500 has been processed and is available in your account.",
        "Investment portfolio quarterly report: Your account has grown by 8.5% this quarter.",
        "Retirement account contribution limit has increased to $23,000 for the upcoming tax year.",
        "Your loan application has been approved. Please contact us to finalize the terms.",
        "Tax document (1099-INT) for interest earned in 2023 is now available for download.",
        "Fraud alert: We noticed unusual activity. Please verify recent transactions on your account.",
        
        # E-commerce/Shopping
        "Your order #ORD-12345 has been shipped and will arrive within 2-3 business days.",
        "Thank you for your purchase! Your order confirmation and tracking information attached.",
        "Price drop alert: The item in your wishlist is now 25% off. Limited time offer!",
        "Your return has been processed. Refund of $89.99 will appear in 3-5 business days.",
        "New arrivals in your favorite category: Women's Athletic Wear. Shop the collection now!",
        "Your subscription to Premium Service has been renewed for another year. Thank you!",
        "Product review request: How did you like the laptop you purchased last month?",
        "Flash sale starting tomorrow: Up to 50% off selected electronics. Members get early access!",
        "Your warranty registration for Model XYZ-123 has been confirmed. Keep this for records.",
        "Loyalty program update: You've earned enough points for a free shipping upgrade!",
        
        # Healthcare/Medical
        "Appointment reminder: Annual check-up scheduled for next Tuesday at 3:30 PM with Dr. Smith.",
        "Your prescription for blood pressure medication is ready for pickup at the pharmacy.",
        "Lab results from your recent blood work are normal. Full report available in patient portal.",
        "Annual mammography screening is due. Please call to schedule your appointment.",
        "Flu shot clinic will be held next Saturday from 9 AM to 4 PM. No appointment necessary.",
        "Insurance claim for your recent visit has been processed. Your portion is $25 copay.",
        "Reminder: Physical therapy session tomorrow at 10 AM. Please arrive 15 minutes early.",
        "Your dermatology appointment has been rescheduled to next Friday at 2 PM due to emergency.",
        "Dental cleaning reminder: You're due for your six-month cleaning and examination.",
        "Vision insurance benefits reset January 1st. Schedule your annual eye exam soon.",
        
        # News/Newsletters
        "Weekly tech newsletter: AI breakthrough, new smartphone releases, and cybersecurity tips.",
        "Local weather alert: Heavy rain expected tomorrow. Plan for potential traffic delays.",
        "Community newsletter: New park opening, school board elections, and upcoming events.",
        "Your favorite restaurant has a new seasonal menu featuring farm-to-table ingredients.",
        "Book club newsletter: February selection announced, author visit scheduled for March.",
        "City council meeting minutes from last week's session are now available online.",
        "Environmental update: Recycling program expansion and new sustainability initiatives announced.",
        "Sports newsletter: Local team wins championship, season schedule updates, and player trades.",
        "Art museum newsletter: New exhibition opening, member preview event, and educational programs.",
        "Neighborhood watch update: Recent safety tips, crime statistics, and community meeting dates."
    ]
    
    # Create balanced dataset
    all_emails = spam_emails + ham_emails
    all_labels = [1] * len(spam_emails) + [0] * len(ham_emails)
    
    print(f"Created enhanced sample dataset: {len(spam_emails)} spam, {len(ham_emails)} ham emails")
    return all_emails, all_labels

def create_production_model():
    """Create a production-ready spam detection model"""
    print("Creating production-ready spam detection model...")
    
    # Try to load real datasets first
    emails, labels = load_enron_dataset()
    
    # If no real dataset available, use enhanced samples
    if emails is None:
        emails, labels = create_enhanced_sample_data()
    
    # Convert to numpy arrays
    emails = np.array(emails)
    labels = np.array(labels)
    
    print(f"Total emails: {len(emails)}")
    print(f"Spam emails: {np.sum(labels)}")
    print(f"Ham emails: {len(labels) - np.sum(labels)}")
    
    # Create more sophisticated preprocessing pipeline
    vectorizer = TfidfVectorizer(
        max_features=10000,           # Increased feature count
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 3),          # Include trigrams
        min_df=2,                    # Ignore terms that appear in less than 2 documents
        max_df=0.95,                 # Ignore terms that appear in more than 95% of documents
        sublinear_tf=True,           # Use sublinear scaling
        strip_accents='unicode'      # Remove accents
    )
    
    # Use more sophisticated model
    model = MultinomialNB(
        alpha=0.1,                   # Smoothing parameter
        fit_prior=True
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # Fit vectorizer on training data
    print("Vectorizing text data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train the model
    print("Training the model...")
    model.fit(X_train_vec, y_train)
    
    # Cross-validation for more robust evaluation
    print("Performing cross-validation...")
    X_all_vec = vectorizer.transform(emails)
    cv_scores = cross_val_score(model, X_all_vec, labels, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Evaluate on test set
    train_predictions = model.predict(X_train_vec)
    test_predictions = model.predict(X_test_vec)
    
    print(f"\nTraining Accuracy: {accuracy_score(y_train, train_predictions):.3f}")
    print(f"Testing Accuracy: {accuracy_score(y_test, test_predictions):.3f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_predictions, 
                              target_names=['Ham (Not Spam)', 'Spam']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives (Ham correctly identified): {cm[0,0]}")
    print(f"False Positives (Ham incorrectly marked as Spam): {cm[0,1]}")
    print(f"False Negatives (Spam missed): {cm[1,0]}")
    print(f"True Positives (Spam correctly identified): {cm[1,1]}")
    
    # Feature importance analysis
    print("\nTop 20 Spam Indicators (words most associated with spam):")
    feature_names = vectorizer.get_feature_names_out()
    spam_features = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    top_spam_indices = spam_features.argsort()[-20:][::-1]
    
    for i, idx in enumerate(top_spam_indices):
        print(f"{i+1:2d}. {feature_names[idx]:<20} (score: {spam_features[idx]:.3f})")
    
    # Save the enhanced model
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'training_accuracy': accuracy_score(y_train, train_predictions),
        'testing_accuracy': accuracy_score(y_test, test_predictions),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_count': len(feature_names),
        'training_samples': len(X_train)
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nProduction model saved as 'model.pkl'")
    print(f"Features: {len(feature_names):,}")
    print(f"Training samples: {len(X_train):,}")
    
    # Test with challenging examples
    print("\n" + "="*60)
    print("TESTING WITH CHALLENGING EXAMPLES:")
    print("="*60)
    
    challenging_tests = [
        ("Congratulations! You have won our weekly prize draw. Please click here to claim.", "should be spam"),
        ("Hi John, congratulations on your promotion! Well deserved.", "should be ham"),
        ("URGENT: Your account needs verification immediately or will be closed!", "should be spam"), 
        ("Urgent: Please review the attached contract by end of day.", "should be ham"),
        ("Free trial of our premium software! No credit card required!", "should be spam"),
        ("The free trial period for your software subscription ends next week.", "should be ham"),
        ("You have been selected for an exclusive investment opportunity!", "should be spam"),
        ("You have been selected for the leadership development program.", "should be ham"),
    ]
    
    for text, expected in challenging_tests:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        
        predicted_label = "spam" if prediction == 1 else "ham"
        confidence = max(probabilities) * 100
        
        print(f"\nText: {text}")
        print(f"Expected: {expected} | Predicted: {predicted_label} | Confidence: {confidence:.1f}%")
        
        # Show if this is a correct prediction
        correct = (prediction == 1 and "spam" in expected) or (prediction == 0 and "ham" in expected)
        print(f"Result: {'✓ CORRECT' if correct else '✗ INCORRECT'}")

if __name__ == "__main__":
    create_production_model()