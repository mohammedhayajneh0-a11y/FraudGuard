# FraudGuard

## Overview
FraudGuard is a fraud detection project focused on identifying suspicious transactions using behavioral patterns.  
The analysis highlights a critical risk window between account signup and the first purchase, where fraud is most likely to occur.

---

## Key Insights

### 1. Time-Based Fraud Behavior
Fraud risk is highest when a transaction occurs shortly after account creation.  
Transactions within the first hour show a significantly higher likelihood of being fraudulent.  
As the time gap increases, fraud probability decreases sharply.

---

### 2. Purchase Value
Purchase value is not a reliable indicator of fraud.  
Fraudulent transactions closely resemble normal user spending behavior, indicating that attackers intentionally mimic legitimate activity.

---

### 3. Geographic Patterns
Fraud rates vary across countries, but location alone is not a strong predictor.  

The highest risk appears when combining:
- Country  
- Fast transaction (< 1 hour)

---

### 4. Behavioral Patterns
Fraud in this dataset is primarily behavior-driven rather than attribute-driven:

- Transaction speed is the strongest signal  
- Demographics (age, gender) have minimal impact  
- Browser and traffic source show limited influence  

---

## Executive Summary
The dominant fraud pattern is:

 Immediate purchase after account signup (within 1 hour)

Time-based features significantly outperform all other variables in detecting fraud.  
Traditional indicators such as purchase value or user demographics provide little predictive power.

---

## Recommendations
- Implement real-time monitoring for transactions within the first hour  
- Add verification steps (OTP / 2FA) for high-risk transactions  
- Focus on high-risk combinations (country + fast transaction)  
- Avoid relying on weak indicators such as purchase value alone  

---

## Technologies
- Python
- Pandas
- Jupyter Notebook
- VS Code

---

## Project Structure
- `main.ipynb` → Main analysis notebook  

---

## Author
Mohammad Hadi Hayajneh
