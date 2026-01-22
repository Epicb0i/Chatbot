"""
Your Friendly Mental Health Support Buddy 💙
- Crisis-aware and empathetic
- Talks like a real friend
- Here for you anytime
"""

import pandas as pd
import numpy as np
import random
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class MentalHealthBuddy:
    def __init__(self, csv_path):
        """Your friendly mental health companion"""
        print("🌟 Getting ready to chat with you...")
        
        self.df = pd.read_csv(csv_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Crisis keywords
        self.crisis_keywords = [
            'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
            'better off dead', 'no reason to live', 'end it all', 'hurt myself',
            'harm myself', 'goodbye letter', 'plan to die', 'ready to go',
            'wish i was dead', 'don\'t want to live', 'ready to end', 'take my life'
        ]
        
        # Separate crisis vs regular
        self.crisis_df = self.df[self.df['Question_ID'] > 100].copy()
        self.general_df = self.df[self.df['Question_ID'] <= 100].copy()
        
        # Process questions
        self.general_df['processed'] = self.general_df['Questions'].apply(self.preprocess)
        
        # TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.general_df['processed'])
        
        print(f"✨ Ready to be your support buddy! ({len(self.general_df)} topics I can help with)")
        print("💙 Let's chat!\n")
    
    def preprocess(self, text):
        """Clean text"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w and len(w) > 2]
        return ' '.join(words)
    
    def is_crisis(self, text):
        """Check if someone needs immediate help"""
        text_lower = text.lower()
        
        # Check keywords
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                return True
        
        # Check patterns
        crisis_patterns = [
            r'\bi want to (die|kill myself|end (it|my life))\b',
            r'\bgoing to (kill myself|die|end it)\b',
            r'\b(better off dead|no point living|can\'t go on)\b',
            r'\bsuicide\b',
            r'\bhurt myself\b'
        ]
        
        for pattern in crisis_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def crisis_support(self):
        """Immediate crisis help"""
        return (
            "🚨 Hey, I'm really worried about you right now. Please, please reach out for help:\n\n"
            "📞 **Call 988** - Suicide & Crisis Lifeline (free, 24/7, confidential)\n"
            "💬 **Text HELLO to 741741** - Crisis Text Line (if calling feels hard)\n"
            "🏥 **Go to the ER** if you're in immediate danger\n\n"
            "I know it's hard, but you don't have to face this alone. "
            "There are people who care and want to help you through this. "
            "Please reach out right now. Your life matters. 💙"
        )
    
    def get_response(self, user_input):
        """Get friendly, supportive response"""
        
        # Crisis check first!
        if self.is_crisis(user_input):
            return self.crisis_support()
        
        user_lower = user_input.lower().strip()
        
        # Greetings
        if user_lower in ['hi', 'hello', 'hey', 'sup', 'yo', 'heya', 'hola']:
            return random.choice([
                "Hey there friend! 😊 How are you doing today? Want to talk about something?",
                "Hi! 💙 I'm here for you. What's on your mind?",
                "Hey! So glad you're here. How are you feeling? Want to chat about anything?",
                "Hello friend! I'm all ears. What's going on with you?"
            ])
        
        # Thanks
        if any(word in user_lower for word in ['thank', 'thanks', 'thx', 'appreciate']):
            return random.choice([
                "Aw, you're so welcome! 💚 I'm always here if you need me, okay?",
                "Of course! That's what friends are for. 😊 Take care!",
                "Anytime! Seriously, I'm here whenever you need to talk. 💙",
                "You got it! Remember, you're not alone in this. 💜"
            ])
        
        # How are you
        if any(phrase in user_lower for phrase in ['how are you', 'how r u', 'hows it going']):
            return random.choice([
                "Thanks for asking! I'm here and ready to support you. More importantly though - how are YOU doing? 😊",
                "I'm good! But let's focus on you - how are you really feeling?",
                "I'm doing well! But I'm more interested in you. What's going on in your world?"
            ])
        
        # Process input
        processed = self.preprocess(user_input)
        
        if not processed or len(processed.split()) < 1:
            return "I'm listening! Tell me more - what's going on? 💙"
        
        # Find match
        user_vector = self.vectorizer.transform([processed])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        best_idx = similarities.argmax()
        best_score = similarities[0][best_idx]
        
        # Good match
        if best_score > 0.15:
            answer = self.general_df.iloc[best_idx]['Answers']
            return self.make_friendly(answer, user_input)
        else:
            # Empathetic fallback
            return self.empathetic_fallback(user_input)
    
    def make_friendly(self, answer, user_input):
        """Make response super friendly"""
        
        # Add friendly openers based on emotion
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['lonely', 'alone', 'isolated', 'no friends']):
            openers = [
                "I'm sorry you're feeling lonely. That really sucks. 💙 ",
                "Loneliness is so hard. I'm here with you. ",
                "Hey, you're not alone - I'm here. And here's what might help: "
            ]
        elif any(word in user_lower for word in ['sad', 'depressed', 'down', 'hopeless']):
            openers = [
                "I hear you, and I'm really sorry you're feeling this way. 💙 ",
                "That sounds so heavy. I'm here for you. ",
                "Ugh, that must be really tough. Let me try to help: "
            ]
        elif any(word in user_lower for word in ['anxious', 'anxiety', 'worried', 'stress', 'overwhelm']):
            openers = [
                "Anxiety is exhausting, I get it. 💚 ",
                "That overwhelm is real. Let's tackle this together: ",
                "I hear you. That anxiety must be draining. Here's what might help: "
            ]
        elif any(word in user_lower for word in ['angry', 'mad', 'frustrated', 'pissed']):
            openers = [
                "I totally hear that frustration. ",
                "It's okay to be angry. That's valid. ",
                "Sounds frustrating for real. "
            ]
        else:
            openers = ["", "I hear you. ", "Okay so, "]
        
        response = random.choice(openers) + answer
        
        # Make more casual
        response = response.replace("It is important", "It's important")
        response = response.replace("It's important to", "You should definitely")
        response = response.replace("individuals", "people")
        response = response.replace("We encourage", "I'd suggest")
        response = response.replace("consider seeking", "talk to")
        
        # Add encouraging endings sometimes
        if len(response) < 200 and random.random() > 0.6:
            endings = [
                " You've got this! 💪",
                " I believe in you!",
                " One step at a time, okay?",
                " You're stronger than you think! ✨"
            ]
            response += random.choice(endings)
        
        return response
    
    def empathetic_fallback(self, user_input):
        """When no match, still be supportive"""
        
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['feel', 'feeling']):
            return random.choice([
                "I hear you. Your feelings are totally valid. Want to tell me more about what's going on? 💙",
                "Thanks for sharing how you're feeling with me. That takes courage. Tell me more?",
                "I'm here to listen. What's been making you feel this way?"
            ])
        
        elif any(word in user_lower for word in ['help', 'what should', 'what can', 'how do i']):
            return random.choice([
                "I want to help! Can you tell me a bit more about what's going on? Then I can give better advice. 😊",
                "Let's figure this out together. What's the main thing you're struggling with?",
                "I'm here for you! Give me some more details about your situation?"
            ])
        
        elif any(word in user_lower for word in ['tired', 'exhausted', 'drained']):
            return random.choice([
                "Being tired all the time is really hard. You deserve rest. What's been draining your energy?",
                "Exhaustion is real. It sounds like you need some serious rest. Tell me more?",
                "I hear that fatigue. Your body might be telling you something. Want to talk about it?"
            ])
        
        else:
            return random.choice([
                "I'm here to listen, friend. Want to tell me more about what's on your mind? 💙",
                "I'm all ears! What's been going on with you?",
                "Thanks for opening up. I want to understand better - can you share more?",
                "I'm here for you. What's been weighing on you lately?"
            ])
    
    def chat(self):
        """Super friendly chat interface"""
        print("\n" + "=" * 70)
        print("💙 Your Mental Health Support Buddy")
        print("=" * 70)
        print("\nHey friend! I'm so glad you're here. 😊")
        print("\nThink of me as that friend who's always there to listen,")
        print("doesn't judge, and genuinely cares about how you're doing.")
        print("\nYou can talk to me about ANYTHING:")
        print("  💭 Feeling lonely, sad, or anxious?")
        print("  😰 Stressed about work or life?")
        print("  🤔 Just need someone to listen?")
        print("  💔 Going through a tough time?")
        print("\nI'm here for all of it. No judgment, just support.")
        print("\n🚨 If you're in crisis, I'll help connect you to immediate support.")
        print("💬 Type 'bye' whenever you need to go. No pressure!\n")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n😊 You: ").strip()
                
                if not user_input:
                    print("🤖 Friend: I'm here. Take your time. 💙")
                    continue
                
                # Exit
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye', 'stop', 'gotta go']:
                    farewells = [
                        "Take care of yourself, okay? I'm always here if you need me. 💙",
                        "Bye friend! Remember - you're doing better than you think. Be kind to yourself! 💚",
                        "See you later! Don't hesitate to come back anytime. You matter! 💜",
                        "Take care! You're stronger than you know. I'm here whenever you need. ✨",
                        "Bye! Remember to be gentle with yourself. You've got this! 💪💙"
                    ]
                    print(f"\n🤖 Friend: {random.choice(farewells)}\n")
                    print("=" * 70)
                    break
                
                # Get response
                response = self.get_response(user_input)
                print(f"\n🤖 Friend: {response}")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖 Friend: Take care! You're important and you matter. Come back anytime! 💙\n")
                print("=" * 70)
                break

def main():
    csv_path = "Chatbot-for-mental-health/Dataset/adult_mental_health.csv"
    
    try:
        buddy = MentalHealthBuddy(csv_path)
        buddy.chat()
    except FileNotFoundError:
        print(f"❌ Oops! Couldn't find the file: {csv_path}")
    except Exception as e:
        print(f"❌ Something went wrong: {e}")

if __name__ == '__main__':
    main()
