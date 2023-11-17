from sklearn.metrics import accuracy_score, classification_report

def evaluate_detection(classifier, X_train, y_train, X_test, y_test, target_names):

    classifier.fit(X_train, y_train)

    yhat = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, yhat)
    report = classification_report(y_test, yhat, target_names=target_names)
    return accuracy, report
