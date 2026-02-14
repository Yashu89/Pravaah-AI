# Design Document: Pravaah AI – Bharat Content Intelligence Engine

## 1. High-Level Architecture

### Architecture Overview

Pravaah AI follows a layered microservices architecture designed for scalability, maintainability, and performance.

**Layer 1: Presentation Layer**
- Next.js/React web dashboard for desktop users
- Responsive design with TailwindCSS
- Real-time updates via WebSocket connections
- Progressive Web App (PWA) capabilities

**Layer 2: API Gateway Layer**
- Centralized API Gateway with load balancing
- JWT-based authentication and authorization
- Rate limiting (100 req/min for creators, 50 req/min for viewers)
- Request routing and service discovery
- API versioning support

**Layer 3: Application Layer (FastAPI Services)**
- Engagement Simulator Service
- Content Time Machine Service
- Photo Optimizer Service
- Misleading Detection Service
- Notification Service
- User Management Service

**Layer 4: AI/ML Layer**
- LLM Engine (OpenAI GPT-4 / Google Gemini)
- Sentiment Analysis Model (fine-tuned BERT)
- Emotion Classifier (multi-label classification)
- Vision Model (ResNet50 + custom aesthetic scoring)
- Fact Verification Engine
- Trend Analysis Engine

**Layer 5: Data Layer**
- MongoDB for document storage (users, posts, reports)
- Redis for caching and session management
- AWS S3 for image/video storage
- CloudFront CDN for content delivery

**Layer 6: External Integration Layer**
- News API for fact-checking
- Google Trends API for trend analysis
- Social media APIs for engagement data


### Technology Stack

**Frontend:**
- Framework: Next.js 14 (React 18)
- Styling: TailwindCSS 3.x
- State Management: Zustand
- Charts: Recharts / Chart.js
- HTTP Client: Axios
- Real-time: Socket.io-client

**Backend:**
- Framework: FastAPI (Python 3.11+)
- Async Runtime: asyncio + uvicorn
- API Documentation: OpenAPI/Swagger
- Validation: Pydantic v2
- Task Queue: Celery + Redis

**AI/ML:**
- LLM: OpenAI GPT-4 API / Google Gemini API
- NLP: Hugging Face Transformers (BERT, RoBERTa)
- Computer Vision: OpenCV, PIL, torchvision
- Image Models: ResNet50, MTCNN (face detection)
- ML Framework: PyTorch 2.0
- Image Processing: scikit-image

**Database:**
- Primary: MongoDB 7.0 (document store)
- Cache: Redis 7.2 (in-memory)
- Object Storage: AWS S3
- CDN: CloudFront

**Infrastructure:**
- Cloud: AWS (EC2, Lambda, S3, CloudFront)
- Containerization: Docker
- Orchestration: Docker Compose (MVP), Kubernetes (production)
- CI/CD: GitHub Actions
- Monitoring: Prometheus + Grafana

**External APIs:**
- News API (newsapi.org)
- Google Trends API
- Twitter API v2 (trending topics)
- Fact-checking APIs (Snopes, FactCheck.org)


---

## 2. System Components Design

### 2.1 Engagement Simulator Engine

The Engagement Simulator predicts how well content will perform before posting.

#### Architecture Flow

```
User Input (Caption + Media) 
  → Content Preprocessor 
  → NLP Pipeline 
  → Emotion Scoring 
  → Controversy Detection 
  → Trend Alignment 
  → Viral Score Calculator 
  → Response with Predictions
```

#### NLP Pipeline

**1. Text Preprocessing**
- Tokenization using BERT tokenizer
- Language detection (Hindi, English, Hinglish)
- Emoji extraction and sentiment mapping
- Hashtag extraction and trend matching

**2. Sentiment Analysis**
- Model: Fine-tuned BERT (bert-base-multilingual-cased)
- Output: Positive, Negative, Neutral scores (0-1)
- Confidence threshold: 0.75

**3. Emotion Classification**
- Model: Multi-label classifier (6 emotions)
- Emotions: Joy, Anger, Sadness, Fear, Surprise, Disgust
- Output: Probability distribution across emotions

#### Emotion Scoring Formula

```python
emotion_intensity = sum(emotion_probabilities * emotion_weights)

emotion_weights = {
    'joy': 0.8,
    'surprise': 0.7,
    'anger': 0.9,
    'fear': 0.6,
    'sadness': 0.5,
    'disgust': 0.7
}

# Normalized to 0-1 range
emotion_intensity_score = min(emotion_intensity, 1.0)
```


#### Controversy Scoring Formula

```python
controversy_score = (
    0.3 * political_keyword_density +
    0.3 * negative_sentiment_intensity +
    0.2 * polarizing_topic_match +
    0.2 * sensational_language_score
)

# Normalized to 0-1 range
controversy_weight = min(controversy_score, 1.0)
```

**Controversy Indicators:**
- Political keywords: ["election", "government", "protest", "BJP", "Congress"]
- Religious keywords: ["Hindu", "Muslim", "temple", "mosque"]
- Sensational phrases: ["shocking", "exposed", "truth revealed"]

#### Viral Score Calculation Formula

```python
viral_score = (
    0.4 * trend_alignment_score +
    0.3 * emotion_intensity_score +
    0.2 * controversy_weight +
    0.1 * historical_engagement_score
)

# Scale to 0-100
viral_score_percentage = viral_score * 100
```

**Component Breakdown:**

1. **Trend Alignment Score (0-1)**
   ```python
   trend_alignment = (
       0.5 * hashtag_trend_match +
       0.3 * topic_trend_match +
       0.2 * timing_relevance
   )
   ```

2. **Emotion Intensity Score (0-1)**
   - Calculated from emotion classifier output
   - Higher intensity = higher engagement potential

3. **Controversy Weight (0-1)**
   - Moderate controversy boosts engagement
   - Extreme controversy may reduce reach (platform penalties)

4. **Historical Engagement Score (0-1)**
   ```python
   historical_score = min(
       user_avg_engagement / platform_avg_engagement,
       1.0
   )
   ```


#### Output Format

```json
{
  "viral_score": 78.5,
  "breakdown": {
    "trend_alignment": 0.85,
    "emotion_intensity": 0.72,
    "controversy": 0.45,
    "historical_performance": 0.68
  },
  "predictions": {
    "estimated_likes": "5K-8K",
    "estimated_shares": "500-1K",
    "estimated_comments": "200-400",
    "reach_estimate": "50K-80K"
  },
  "recommendations": [
    "Add trending hashtag #AIForBharat",
    "Post between 7-9 PM for maximum reach",
    "Consider reducing controversial language"
  ],
  "sentiment": {
    "positive": 0.75,
    "negative": 0.15,
    "neutral": 0.10
  },
  "emotions": {
    "joy": 0.65,
    "surprise": 0.25,
    "anger": 0.10
  }
}
```

---

### 2.2 Content Time Machine

Analyzes historical content performance and provides optimization recommendations.

#### Architecture Flow

```
User Request 
  → Historical Data Fetcher 
  → Engagement Trend Analyzer 
  → Best Time Predictor 
  → Caption Optimizer 
  → Report Generator
```

#### Historical Data Ingestion

**Data Sources:**
- User's past posts from MongoDB
- Engagement metrics (likes, shares, comments, views)
- Posting timestamps
- Content type (image, video, text)
- Hashtags and captions


**Data Structure:**
```python
{
  "post_id": "uuid",
  "user_id": "uuid",
  "timestamp": "ISO8601",
  "content_type": "image|video|text",
  "caption": "string",
  "hashtags": ["tag1", "tag2"],
  "metrics": {
    "likes": 1500,
    "shares": 200,
    "comments": 50,
    "views": 10000,
    "engagement_rate": 0.175
  }
}
```

#### Engagement Trend Model

**Time-Series Analysis:**
```python
# Group posts by time slots (hourly)
time_slots = {
  "00:00-01:00": avg_engagement,
  "01:00-02:00": avg_engagement,
  ...
  "23:00-00:00": avg_engagement
}

# Day of week analysis
day_performance = {
  "Monday": avg_engagement,
  "Tuesday": avg_engagement,
  ...
  "Sunday": avg_engagement
}

# Content type performance
content_performance = {
  "image": avg_engagement,
  "video": avg_engagement,
  "carousel": avg_engagement
}
```

#### Best Time Prediction Algorithm

```python
def predict_best_time(user_history, global_trends):
    # Weighted combination of user history and platform trends
    user_weight = 0.7
    global_weight = 0.3
    
    best_times = []
    
    for hour in range(24):
        user_score = user_history.get_engagement_at_hour(hour)
        global_score = global_trends.get_engagement_at_hour(hour)
        
        combined_score = (
            user_weight * user_score +
            global_weight * global_score
        )
        
        best_times.append({
            "hour": hour,
            "score": combined_score,
            "day_preference": get_best_day_for_hour(hour)
        })
    
    # Return top 3 time slots
    return sorted(best_times, key=lambda x: x['score'], reverse=True)[:3]
```


#### Caption Optimization Logic

**Analysis Components:**
1. **Length Analysis**
   - Optimal length: 125-150 characters
   - Too short: < 50 characters (low context)
   - Too long: > 200 characters (reduced readability)

2. **Hashtag Optimization**
   ```python
   optimal_hashtag_count = 5-8
   
   hashtag_score = {
       "trending": 0.4,      # Currently trending
       "relevant": 0.3,      # Topic relevance
       "niche": 0.2,         # Audience specificity
       "engagement": 0.1     # Historical performance
   }
   ```

3. **Call-to-Action (CTA) Detection**
   - Presence of CTA increases engagement by 15-20%
   - Examples: "Share your thoughts", "Tag a friend", "Double tap if you agree"

4. **Emoji Usage**
   - Optimal: 2-4 emojis per caption
   - Placement: Beginning or end for maximum impact

**Output Format:**
```json
{
  "analysis": {
    "current_length": 180,
    "optimal_length": "125-150",
    "hashtag_count": 12,
    "optimal_hashtags": "5-8",
    "has_cta": false,
    "emoji_count": 1
  },
  "best_posting_times": [
    {
      "time": "19:00-21:00",
      "day": "Tuesday",
      "expected_engagement": "high",
      "confidence": 0.85
    }
  ],
  "recommendations": [
    "Reduce caption length to 125-150 characters",
    "Remove 4-5 hashtags, keep only trending ones",
    "Add a call-to-action at the end"
  ],
  "optimized_caption": "Your caption with improvements...",
  "suggested_hashtags": ["#AIForBharat", "#TechIndia"]
}
```


---

### 2.3 Smart Post Optimizer

Automatically selects the best photos from a large collection and creates optimized posts.

#### Architecture Flow

```
User Upload (50-60 photos) 
  → Image Preprocessing 
  → Quality Filtering 
  → Duplicate Detection 
  → Face Detection 
  → Aesthetic Scoring 
  → Ranking & Selection 
  → Collage Generation 
  → Final Output (8-10 best photos + collages)
```

#### Image Preprocessing

**1. Image Loading & Validation**
```python
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize for processing (maintain aspect ratio)
    max_dimension = 1024
    height, width = img_rgb.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_rgb = cv2.resize(img_rgb, (new_width, new_height))
    
    return img_rgb
```

#### Blur Detection (Variance of Laplacian)

```python
def detect_blur(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Threshold: < 100 = blurry, > 100 = sharp
    is_blurry = laplacian_var < 100
    blur_score = min(laplacian_var / 500, 1.0)  # Normalize to 0-1
    
    return {
        "is_blurry": is_blurry,
        "blur_score": blur_score,
        "laplacian_variance": laplacian_var
    }
```


#### Duplicate Detection (Perceptual Hashing)

```python
import imagehash
from PIL import Image

def detect_duplicates(image_list, threshold=5):
    hashes = {}
    duplicates = []
    
    for idx, img_path in enumerate(image_list):
        img = Image.open(img_path)
        img_hash = imagehash.phash(img)
        
        # Check for similar images
        for existing_idx, existing_hash in hashes.items():
            if img_hash - existing_hash < threshold:
                duplicates.append((existing_idx, idx))
        
        hashes[idx] = img_hash
    
    return duplicates
```

#### Face Detection

```python
import cv2

def detect_faces(image):
    # Use Haar Cascade or MTCNN for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    return {
        "face_count": len(faces),
        "face_locations": faces.tolist(),
        "has_faces": len(faces) > 0
    }
```

#### Aesthetic Scoring Model

```python
def calculate_aesthetic_score(image, face_info, blur_info):
    # Component scores
    composition_score = analyze_composition(image)
    color_harmony_score = analyze_color_harmony(image)
    lighting_score = analyze_lighting(image)
    
    # Weighted aesthetic score
    aesthetic_score = (
        0.25 * composition_score +
        0.20 * color_harmony_score +
        0.20 * lighting_score +
        0.20 * blur_info['blur_score'] +
        0.15 * (1.0 if face_info['has_faces'] else 0.5)
    )
    
    return min(aesthetic_score, 1.0)

def analyze_composition(image):
    # Rule of thirds analysis
    height, width = image.shape[:2]
    third_h, third_w = height // 3, width // 3
    
    # Detect edges and check if they align with rule of thirds
    edges = cv2.Canny(image, 100, 200)
    
    # Calculate edge density in key regions
    roi_score = calculate_roi_density(edges, third_h, third_w)
    
    return roi_score

def analyze_color_harmony(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate color distribution
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
    # Normalize and calculate variance
    hist_norm = hist / hist.sum()
    variance = np.var(hist_norm)
    
    # Lower variance = better color harmony
    harmony_score = 1.0 - min(variance * 10, 1.0)
    
    return harmony_score
```


#### Ranking Algorithm

```python
def rank_images(images_with_scores):
    # Final ranking formula
    for img in images_with_scores:
        final_score = (
            0.40 * img['aesthetic_score'] +
            0.25 * img['blur_score'] +
            0.20 * (1.0 if img['has_faces'] else 0.3) +
            0.15 * img['uniqueness_score']  # Based on duplicate detection
        )
        img['final_score'] = final_score
    
    # Sort by final score
    ranked = sorted(images_with_scores, key=lambda x: x['final_score'], reverse=True)
    
    # Select top 8-10 images
    return ranked[:10]
```

#### Collage Generation Logic

```python
def generate_collage(selected_images, layout="grid"):
    """
    Creates optimized collages from selected images
    
    Layouts:
    - grid: 2x2, 3x3 grids
    - horizontal: Side-by-side panorama
    - vertical: Stacked layout
    - mixed: Best images in larger tiles
    """
    
    if layout == "grid":
        # Create 3x3 grid collage
        collage = create_grid_collage(selected_images[:9], rows=3, cols=3)
    elif layout == "horizontal":
        collage = create_horizontal_collage(selected_images[:4])
    elif layout == "mixed":
        # Highlight best image, smaller tiles for others
        collage = create_mixed_layout(selected_images[:6])
    
    return collage
```

**Output Format:**
```json
{
  "selected_photos": [
    {
      "photo_id": "img_001",
      "score": 0.92,
      "reasons": ["High aesthetic quality", "Clear faces", "Good lighting"]
    }
  ],
  "collages": [
    {
      "collage_id": "collage_001",
      "layout": "grid_3x3",
      "url": "s3://bucket/collages/collage_001.jpg",
      "photos_used": ["img_001", "img_003", "img_007"]
    }
  ],
  "rejected_photos": [
    {
      "photo_id": "img_015",
      "reason": "Blurry (score: 0.32)"
    }
  ]
}
```


---

### 2.4 Misleading Content Detection Engine

Detects and flags misleading or fake content through community validation and AI verification.

#### Architecture Flow

```
User Reports Content 
  → Claim Extraction (LLM) 
  → Sensational Keyword Detection 
  → Fact Verification (External APIs) 
  → Risk Score Calculation 
  → Community Validation Queue 
  → Decision & Alert Propagation
```

#### Claim Extraction via LLM

```python
def extract_claims(content_text, content_media):
    """
    Uses LLM to extract factual claims from content
    """
    prompt = f"""
    Analyze the following social media content and extract all factual claims:
    
    Text: {content_text}
    
    For each claim, provide:
    1. The claim statement
    2. Whether it's verifiable
    3. Key entities mentioned
    4. Context needed for verification
    
    Format as JSON.
    """
    
    response = llm_api.generate(prompt)
    claims = parse_claims(response)
    
    return claims
```

**Example Output:**
```json
{
  "claims": [
    {
      "statement": "India won the war in 2024",
      "verifiable": true,
      "entities": ["India", "war", "2024"],
      "context": "Military conflict"
    }
  ]
}
```

#### Sensational Keyword Scoring

```python
def calculate_sensational_score(text):
    sensational_keywords = [
        "shocking", "exposed", "truth revealed", "they don't want you to know",
        "breaking", "urgent", "must watch", "viral", "unbelievable"
    ]
    
    text_lower = text.lower()
    keyword_count = sum(1 for kw in sensational_keywords if kw in text_lower)
    
    # Normalize by text length
    sensational_density = keyword_count / max(len(text.split()), 1)
    
    # Score from 0-1
    sensational_score = min(sensational_density * 10, 1.0)
    
    return sensational_score
```


#### Fact Comparison Logic

```python
def verify_claims(claims, news_api, fact_check_api):
    """
    Cross-references claims with trusted sources
    """
    verification_results = []
    
    for claim in claims:
        # Search news databases
        news_results = news_api.search(claim['statement'])
        
        # Search fact-checking databases
        fact_check_results = fact_check_api.search(claim['statement'])
        
        # Calculate match score
        match_score = calculate_match_score(
            claim, news_results, fact_check_results
        )
        
        verification_results.append({
            "claim": claim['statement'],
            "match_score": match_score,
            "sources": news_results[:3],
            "fact_checks": fact_check_results[:2]
        })
    
    return verification_results
```

#### Risk Score Calculation

```python
def calculate_misleading_risk(claim_verification, sensational_score, context_analysis):
    """
    Misleading Risk = 
        0.4 * Claim Mismatch +
        0.3 * Sensational Intensity +
        0.3 * Context Inconsistency
    """
    
    # Claim mismatch (0-1): How much claims contradict verified facts
    claim_mismatch = 1.0 - np.mean([v['match_score'] for v in claim_verification])
    
    # Sensational intensity (0-1): Already calculated
    sensational_intensity = sensational_score
    
    # Context inconsistency (0-1): Temporal, geographical, logical inconsistencies
    context_inconsistency = analyze_context_inconsistency(context_analysis)
    
    # Final risk score
    risk_score = (
        0.4 * claim_mismatch +
        0.3 * sensational_intensity +
        0.3 * context_inconsistency
    )
    
    return min(risk_score, 1.0)
```

#### Community Validation Workflow

```python
def community_validation_process(report):
    """
    Multi-stage validation process
    """
    # Stage 1: Initial AI screening
    ai_risk_score = calculate_misleading_risk(...)
    
    if ai_risk_score < 0.3:
        return {"status": "low_risk", "action": "no_action"}
    
    # Stage 2: Community review (if risk > 0.3)
    if ai_risk_score >= 0.3:
        # Assign to trusted reviewers
        reviewers = select_trusted_reviewers(count=3)
        
        # Collect votes
        votes = collect_reviewer_votes(reviewers, report)
        
        # Calculate consensus
        consensus = calculate_consensus(votes)
        
        if consensus >= 0.67:  # 2 out of 3 agree
            return {
                "status": "misleading_confirmed",
                "action": "flag_and_notify",
                "consensus": consensus
            }
    
    return {"status": "inconclusive", "action": "escalate"}
```


#### Alert Propagation Mechanism

```python
def propagate_alerts(misleading_post_id, affected_users):
    """
    Notifies all users who viewed the misleading content
    """
    notification_payload = {
        "type": "misleading_content_alert",
        "post_id": misleading_post_id,
        "message": "A post you viewed has been flagged as misleading",
        "severity": "high",
        "action_required": "review_content"
    }
    
    # Send push notifications
    for user_id in affected_users:
        send_push_notification(user_id, notification_payload)
        
        # Log notification in database
        log_notification(user_id, notification_payload)
    
    # Update post status
    update_post_status(misleading_post_id, status="flagged_misleading")
```

**Output Format:**
```json
{
  "report_id": "rpt_12345",
  "content_id": "post_67890",
  "risk_score": 0.78,
  "risk_breakdown": {
    "claim_mismatch": 0.85,
    "sensational_intensity": 0.65,
    "context_inconsistency": 0.72
  },
  "claims_analyzed": [
    {
      "claim": "India won the war in 2024",
      "verification_status": "false",
      "sources": [
        {
          "title": "No such war occurred",
          "url": "https://factcheck.org/...",
          "credibility": 0.95
        }
      ]
    }
  ],
  "community_validation": {
    "status": "misleading_confirmed",
    "consensus": 0.67,
    "reviewer_count": 3
  },
  "action_taken": "flagged_and_notified",
  "users_notified": 15420
}
```

---

## 3. Database Schema Design

### 3.1 Users Collection

```javascript
{
  "_id": ObjectId,
  "user_id": "uuid",
  "username": "string",
  "email": "string",
  "password_hash": "string",
  "role": "creator|viewer|moderator",
  "profile": {
    "display_name": "string",
    "bio": "string",
    "avatar_url": "string",
    "followers_count": 0,
    "following_count": 0
  },
  "reputation_score": 100,
  "engagement_stats": {
    "total_posts": 0,
    "avg_engagement_rate": 0.0,
    "total_likes": 0,
    "total_shares": 0
  },
  "preferences": {
    "language": "en|hi",
    "notification_settings": {}
  },
  "created_at": ISODate,
  "updated_at": ISODate,
  "last_login": ISODate
}
```


### 3.2 Posts Collection

```javascript
{
  "_id": ObjectId,
  "post_id": "uuid",
  "user_id": "uuid",
  "content_type": "image|video|text|carousel",
  "caption": "string",
  "hashtags": ["tag1", "tag2"],
  "media_urls": ["s3://bucket/image1.jpg"],
  "viral_score": 78.5,
  "viral_breakdown": {
    "trend_alignment": 0.85,
    "emotion_intensity": 0.72,
    "controversy": 0.45,
    "historical_performance": 0.68
  },
  "engagement": {
    "likes": 0,
    "shares": 0,
    "comments": 0,
    "views": 0,
    "engagement_rate": 0.0
  },
  "status": "draft|published|flagged_misleading",
  "posted_at": ISODate,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

### 3.3 Engagement Analysis Collection

```javascript
{
  "_id": ObjectId,
  "analysis_id": "uuid",
  "user_id": "uuid",
  "analysis_type": "time_machine|simulator",
  "time_period": {
    "start_date": ISODate,
    "end_date": ISODate
  },
  "insights": {
    "best_posting_times": [
      {
        "time": "19:00-21:00",
        "day": "Tuesday",
        "expected_engagement": "high",
        "confidence": 0.85
      }
    ],
    "top_performing_content_types": ["video", "carousel"],
    "avg_engagement_by_hour": {},
    "hashtag_performance": {}
  },
  "recommendations": ["string"],
  "created_at": ISODate
}
```

### 3.4 Misleading Reports Collection

```javascript
{
  "_id": ObjectId,
  "report_id": "uuid",
  "post_id": "uuid",
  "reported_by": "uuid",
  "report_reason": "string",
  "evidence": {
    "description": "string",
    "supporting_links": ["url1", "url2"]
  },
  "ai_analysis": {
    "risk_score": 0.78,
    "risk_breakdown": {
      "claim_mismatch": 0.85,
      "sensational_intensity": 0.65,
      "context_inconsistency": 0.72
    },
    "claims_analyzed": []
  },
  "community_validation": {
    "status": "pending|confirmed|rejected",
    "reviewer_votes": [],
    "consensus": 0.67
  },
  "action_taken": "none|flagged|removed",
  "users_notified": 0,
  "status": "pending|under_review|resolved",
  "created_at": ISODate,
  "resolved_at": ISODate
}
```

### 3.5 Reputation Scores Collection

```javascript
{
  "_id": ObjectId,
  "user_id": "uuid",
  "reputation_score": 100,
  "score_history": [
    {
      "date": ISODate,
      "score": 100,
      "change": 0,
      "reason": "string"
    }
  ],
  "contributions": {
    "accurate_reports": 0,
    "false_reports": 0,
    "community_reviews": 0
  },
  "badges": ["trusted_reviewer", "fact_checker"],
  "updated_at": ISODate
}
```


---

## 4. API Design

### 4.1 Engagement Simulator API

**Endpoint:** `POST /api/v1/simulate`

**Request:**
```json
{
  "user_id": "uuid",
  "caption": "Excited to share my AI journey! #AIForBharat #TechIndia",
  "media_urls": ["https://s3.../image1.jpg"],
  "content_type": "image",
  "scheduled_time": "2024-03-15T19:30:00Z"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "viral_score": 78.5,
    "breakdown": {
      "trend_alignment": 0.85,
      "emotion_intensity": 0.72,
      "controversy": 0.45,
      "historical_performance": 0.68
    },
    "predictions": {
      "estimated_likes": "5K-8K",
      "estimated_shares": "500-1K",
      "estimated_comments": "200-400",
      "reach_estimate": "50K-80K"
    },
    "recommendations": [
      "Add trending hashtag #Innovation",
      "Post between 7-9 PM for maximum reach"
    ]
  }
}
```

---

### 4.2 Photo Optimizer API

**Endpoint:** `POST /api/v1/optimize`

**Request:**
```json
{
  "user_id": "uuid",
  "photo_urls": [
    "https://s3.../photo1.jpg",
    "https://s3.../photo2.jpg",
    "... (50-60 photos)"
  ],
  "preferences": {
    "max_photos": 10,
    "include_collages": true,
    "collage_layouts": ["grid", "mixed"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "selected_photos": [
      {
        "photo_id": "img_001",
        "url": "https://s3.../photo1.jpg",
        "score": 0.92,
        "reasons": ["High aesthetic quality", "Clear faces", "Good lighting"]
      }
    ],
    "collages": [
      {
        "collage_id": "collage_001",
        "layout": "grid_3x3",
        "url": "https://s3.../collages/collage_001.jpg",
        "photos_used": ["img_001", "img_003", "img_007"]
      }
    ],
    "rejected_photos": [
      {
        "photo_id": "img_015",
        "reason": "Blurry (score: 0.32)"
      }
    ],
    "processing_time_ms": 4500
  }
}
```

---

### 4.3 Content Time Machine API

**Endpoint:** `POST /api/v1/analyze-history`

**Request:**
```json
{
  "user_id": "uuid",
  "time_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-01"
  },
  "analysis_type": "full"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis_12345",
    "best_posting_times": [
      {
        "time": "19:00-21:00",
        "day": "Tuesday",
        "expected_engagement": "high",
        "confidence": 0.85
      }
    ],
    "caption_analysis": {
      "avg_length": 145,
      "optimal_length": "125-150",
      "avg_hashtags": 8,
      "optimal_hashtags": "5-8"
    },
    "content_performance": {
      "best_type": "video",
      "engagement_by_type": {
        "image": 0.12,
        "video": 0.18,
        "carousel": 0.15
      }
    },
    "recommendations": [
      "Post videos on Tuesday evenings",
      "Reduce hashtag count to 5-8",
      "Add more CTAs to captions"
    ]
  }
}
```


---

### 4.4 Misleading Content Check API

**Endpoint:** `POST /api/v1/misleading-check`

**Request:**
```json
{
  "post_id": "uuid",
  "content_text": "Breaking: India won the war in 2024!",
  "media_urls": ["https://s3.../image.jpg"],
  "requester_id": "uuid"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "risk_score": 0.78,
    "risk_level": "high",
    "risk_breakdown": {
      "claim_mismatch": 0.85,
      "sensational_intensity": 0.65,
      "context_inconsistency": 0.72
    },
    "claims_analyzed": [
      {
        "claim": "India won the war in 2024",
        "verification_status": "false",
        "confidence": 0.92
      }
    ],
    "recommendation": "flag_for_review"
  }
}
```

---

### 4.5 Report Content API

**Endpoint:** `POST /api/v1/report-content`

**Request:**
```json
{
  "post_id": "uuid",
  "reported_by": "uuid",
  "report_reason": "misleading_information",
  "evidence": {
    "description": "This post contains false information about a war",
    "supporting_links": [
      "https://factcheck.org/article123",
      "https://newsapi.org/article456"
    ]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "report_id": "rpt_12345",
    "status": "under_review",
    "estimated_review_time": "24-48 hours",
    "message": "Thank you for reporting. Our team will review this content."
  }
}
```

---

### 4.6 Dashboard API

**Endpoint:** `GET /api/v1/dashboard`

**Query Parameters:**
- `user_id`: string (required)
- `time_range`: string (optional, default: "7d")

**Response:**
```json
{
  "success": true,
  "data": {
    "user_stats": {
      "total_posts": 45,
      "avg_viral_score": 72.3,
      "total_engagement": 125000,
      "reputation_score": 850
    },
    "recent_analyses": [
      {
        "analysis_id": "analysis_001",
        "type": "time_machine",
        "created_at": "2024-03-10T10:30:00Z"
      }
    ],
    "trending_topics": [
      "#AIForBharat",
      "#TechIndia",
      "#Innovation"
    ],
    "notifications": [
      {
        "type": "misleading_alert",
        "message": "A post you viewed has been flagged",
        "timestamp": "2024-03-10T09:15:00Z"
      }
    ]
  }
}
```

---

## 5. Scalability Plan

### 5.1 Horizontal Scaling

**Load Balancing Strategy:**
- Use AWS Application Load Balancer (ALB)
- Distribute traffic across multiple FastAPI instances
- Auto-scaling groups based on CPU/memory metrics
- Target: Handle 10,000+ concurrent users

**Database Scaling:**
- MongoDB sharding for horizontal partitioning
- Read replicas for query distribution
- Redis cluster for distributed caching
- Connection pooling to optimize database connections

### 5.2 Microservice Split (Future)

**Service Decomposition:**
```
Current Monolith → Future Microservices:
- User Service (authentication, profiles)
- Engagement Service (simulator, analytics)
- Media Service (photo optimization, storage)
- Detection Service (misleading content)
- Notification Service (alerts, emails)
```

**Benefits:**
- Independent scaling per service
- Technology flexibility
- Fault isolation
- Easier maintenance


### 5.3 Model Caching

**AI Model Optimization:**
- Cache LLM responses for similar queries (Redis)
- Pre-compute embeddings for common hashtags/topics
- Model quantization for faster inference
- Batch processing for image analysis

**Caching Strategy:**
```python
# Cache viral score predictions for similar content
cache_key = f"viral_score:{hash(caption)}:{content_type}"
cached_result = redis.get(cache_key)

if cached_result:
    return cached_result
else:
    result = calculate_viral_score(...)
    redis.setex(cache_key, 3600, result)  # 1 hour TTL
    return result
```

### 5.4 Async Processing

**Background Task Queue (Celery):**
- Photo optimization (long-running)
- Historical data analysis
- Fact-checking verification
- Notification delivery

**Implementation:**
```python
@celery.task
def optimize_photos_async(user_id, photo_urls):
    # Process 50-60 photos in background
    results = photo_optimizer.process(photo_urls)
    
    # Store results in database
    save_optimization_results(user_id, results)
    
    # Notify user when complete
    send_notification(user_id, "Photo optimization complete")
```

**Message Queue Architecture:**
```
FastAPI → Redis Queue → Celery Workers → MongoDB
                ↓
         Task Results
```

---

## 6. Security Design

### 6.1 JWT Authentication

**Token Structure:**
```json
{
  "user_id": "uuid",
  "role": "creator|viewer|moderator",
  "exp": 1234567890,
  "iat": 1234567890
}
```

**Implementation:**
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 6.2 Rate Limiting

**Rate Limit Rules:**
- Creators: 100 requests/minute
- Viewers: 50 requests/minute
- Moderators: 200 requests/minute

**Implementation (Redis-based):**
```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/simulate")
@limiter.limit("100/minute")
async def simulate_engagement(request: Request, data: SimulateRequest):
    # Process request
    pass
```

### 6.3 Role-Based Access Control (RBAC)

**Permission Matrix:**

| Feature | Creator | Viewer | Moderator |
|---------|---------|--------|-----------|
| Engagement Simulator | ✓ | ✗ | ✓ |
| Photo Optimizer | ✓ | ✗ | ✓ |
| Time Machine | ✓ | ✗ | ✓ |
| Report Content | ✓ | ✓ | ✓ |
| Review Reports | ✗ | ✗ | ✓ |
| Manage Users | ✗ | ✗ | ✓ |

**Implementation:**
```python
def require_role(allowed_roles: list):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = get_current_user()
            if user.role not in allowed_roles:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/api/v1/simulate")
@require_role(["creator", "moderator"])
async def simulate_engagement(data: SimulateRequest):
    pass
```


### 6.4 Abuse Prevention

**Anti-Spam Measures:**
1. **Content Fingerprinting**
   - Hash content to detect duplicate reports
   - Block users submitting identical reports repeatedly

2. **Reputation-Based Throttling**
   - Users with low reputation scores face stricter rate limits
   - Trusted users (high reputation) get higher limits

3. **CAPTCHA Integration**
   - Require CAPTCHA for sensitive actions (reporting, flagging)
   - Prevent automated abuse

4. **IP-Based Blocking**
   - Track suspicious IP addresses
   - Temporary bans for repeated violations

**Implementation:**
```python
def check_abuse_score(user_id: str) -> bool:
    """
    Returns True if user is likely abusing the system
    """
    recent_reports = get_recent_reports(user_id, hours=24)
    
    # Check for spam patterns
    if len(recent_reports) > 50:  # Too many reports
        return True
    
    # Check for duplicate content
    unique_content = len(set([r['content_hash'] for r in recent_reports]))
    if unique_content < len(recent_reports) * 0.3:  # 70% duplicates
        return True
    
    return False
```

**Data Encryption:**
- All data in transit: TLS 1.3
- Sensitive data at rest: AES-256 encryption
- Password hashing: bcrypt with salt

---

## 7. Future Enhancements

### 7.1 Government Integration

**Objective:** Partner with government agencies for verified fact-checking

**Features:**
- Direct API integration with PIB (Press Information Bureau)
- Real-time alerts for government-verified information
- Official badge for government-verified content
- Emergency broadcast system for critical updates

**Implementation Approach:**
```
Pravaah AI ←→ Government API Gateway ←→ PIB/Ministry APIs
```

### 7.2 Regional Language Support

**Target Languages:**
- Hindi (Devanagari script)
- Tamil, Telugu, Bengali, Marathi
- Gujarati, Kannada, Malayalam, Punjabi

**Technical Requirements:**
- Multilingual BERT models (IndicBERT)
- Language-specific sentiment analysis
- Regional trend detection
- Localized UI/UX

**Challenges:**
- Code-mixing (Hinglish, Tanglish)
- Script variations
- Cultural context understanding

### 7.3 Real-Time Reel Scanning

**Objective:** Analyze video content in real-time for misleading information

**Features:**
- Frame-by-frame analysis
- Audio transcription and fact-checking
- Deepfake detection
- Context verification

**Technical Stack:**
- Video processing: FFmpeg
- Speech-to-text: Whisper API
- Deepfake detection: CNN-based models
- Real-time processing: WebRTC

**Architecture:**
```
Video Upload → Frame Extraction → Vision Model
                     ↓
              Audio Extraction → Speech-to-Text → LLM Analysis
                     ↓
              Deepfake Detection → Risk Score
```

### 7.4 Browser Extension

**Objective:** Real-time content verification while browsing social media

**Features:**
- Inline fact-checking badges
- Hover tooltips with verification status
- One-click reporting
- Personalized content warnings

**Supported Platforms:**
- Twitter/X
- Facebook
- Instagram
- YouTube

**Technical Implementation:**
- Chrome Extension (Manifest V3)
- Content scripts for DOM manipulation
- Background service worker for API calls
- Local storage for caching

**User Flow:**
```
User views post → Extension detects content → API call to Pravaah
                                                      ↓
                                            Risk score calculated
                                                      ↓
                                            Badge displayed inline
```

### 7.5 Additional Future Features

**AI-Powered Content Suggestions:**
- Auto-generate captions based on images
- Suggest optimal posting schedules
- Recommend trending topics to cover

**Collaborative Fact-Checking:**
- Crowdsourced verification
- Expert reviewer network
- Blockchain-based proof of verification

**Advanced Analytics:**
- Competitor analysis
- Audience demographics insights
- Content performance predictions

**Mobile App:**
- Native iOS and Android apps
- Offline mode for draft creation
- Push notifications for alerts

---

## 8. Monitoring & Observability

### 8.1 Metrics to Track

**Application Metrics:**
- API response times (p50, p95, p99)
- Request throughput (req/sec)
- Error rates by endpoint
- Cache hit/miss ratios

**Business Metrics:**
- Daily active users (DAU)
- Viral score accuracy
- Report resolution time
- User reputation distribution

**Infrastructure Metrics:**
- CPU/Memory utilization
- Database query performance
- S3 storage usage
- CDN bandwidth

### 8.2 Logging Strategy

**Log Levels:**
- ERROR: System failures, exceptions
- WARN: Degraded performance, rate limit hits
- INFO: User actions, API calls
- DEBUG: Detailed execution traces

**Centralized Logging:**
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Structured JSON logs
- Correlation IDs for request tracing

### 8.3 Alerting

**Critical Alerts:**
- API downtime > 1 minute
- Error rate > 5%
- Database connection failures
- S3 upload failures

**Warning Alerts:**
- Response time > 2 seconds
- Cache hit rate < 70%
- Disk usage > 80%

---

## 9. Deployment Strategy

### 9.1 Environment Setup

**Environments:**
1. **Development:** Local Docker Compose
2. **Staging:** AWS EC2 (single instance)
3. **Production:** AWS EKS (Kubernetes cluster)

### 9.2 CI/CD Pipeline

```
GitHub Push → GitHub Actions → Run Tests → Build Docker Image
                                                ↓
                                    Push to ECR → Deploy to EKS
                                                ↓
                                    Health Check → Route Traffic
```

### 9.3 Rollback Strategy

- Blue-Green deployment for zero downtime
- Automated rollback on health check failures
- Database migration versioning
- Feature flags for gradual rollout

---

## Conclusion

This design document provides a comprehensive technical blueprint for Pravaah AI. The architecture is designed to be scalable, secure, and maintainable, with clear paths for future enhancements. The system leverages modern AI/ML technologies while maintaining performance and reliability standards suitable for production deployment.
