![image](https://github.com/user-attachments/assets/905b3189-ad40-4cd9-8b02-e7741510fccc)
![image](https://github.com/user-attachments/assets/255d113b-fefd-4364-bdb2-51f80dec5e9d)


Fake news is a news designed to deliberately spread hoaxes, propaganda and misinformation. 
In today's digital landscape, misinformation poses a significant challenge to news platforms, undermining trust and credibility.
In our efforts to combat fake news effectively, we employed an ensemble learning approach.
Addressing the inherent shortcomings of single-model approaches, such as inaccuracies, limited reliability, and generalization issues.
The core of our proposed system revolves around Ensemble Learning techniques, which combine multiple models for stance detection. By leveraging the collective intelligence of diverse models.

Architecture Diagram:
![image](https://github.com/user-attachments/assets/8de76b84-5ace-40f2-9e56-719dea4345fe)

Text preprocessing is an essential module that ensures the input text data is transformed into a format suitable for further analysis.

Data Cleaning: This step involves the removal of any irrelevant characters, symbols, or HTML tags from the text.

Tokenization: Tokenization breaks down the text into individual words making it more manageable for analysis. Tokens are the basic units of text,
and this step is crucial for further NLP tasks.

Text Normalization: Text normalization includes technique such as lemmatization. It reduces words to their base or root form.

Utilized Tf-Idf Vectorizer, which transforms text into meaningful numerical representations by extracting features based on occurrence.

Built five models – Passive Aggressive Classifier, Multinomial Naivem Bayes, Random Forest, Logistic Regression, and Support Vector Machine – for effective classification.

Dataset is then used to train a model, while the final result is an aggregation of outcome from each model.

Ensemble learning is a machine learning technique that enhances accuracy in fake news detection by merging predictions from multiple models.

In ensemble learning with majority voting, each model's prediction contributes to a "vote” for each test instance.

The final prediction is determined by aggregating these votes, and the class with the majority of votes is selected as the final prediction.

![image](https://github.com/user-attachments/assets/8076b15c-6ed0-4af3-aa25-75c617e05e1d)



