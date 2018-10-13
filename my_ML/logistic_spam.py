from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.naive_bayes import GaussianNB

with open('./SMSSpamCollection.txt',encoding='utf-8') as fr:
    sms_list = fr.readlines()
raw_dataset = np.array(list(map(lambda x:x.split('\t'),sms_list)))
print(raw_dataset)
y = raw_dataset[:,0]=='spam'
print(y)
X_text = raw_dataset[:,1]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_text,y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
print(X_train[0])
clf = LogisticRegression()
clf = GaussianNB()
print(X_train.shape)
clf.fit(np.array(X_train),y_train)
y_hat = clf.predict(X_test)
print(y_hat)
acc_list = (y_hat==y_test)
print('acc',sum(acc_list)/len(acc_list))

my_mail = "Congratulations!! You have been selected to a big prize of $100000! Call 900-222 to know how to get it!"
my_mail2 = "Good news!The worlds biggest online casino has opened! Every day you have a chance to win $10000 CASH! beautiful lotus gambling with you！ our website: www.aomenduchang.com hit it or regreted!"
my_mail3 = "hi darling, what do you want to eat tonight?"
my_mail4 = "Dear Mr.Zhang: We are regreted to inform you that you did not pass our interview this Tuesday. Good luck next time."
my_mail_vector = vectorizer.transform([my_mail])
print(clf.predict(my_mail_vector))


