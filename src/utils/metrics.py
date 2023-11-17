from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

def simple_classification(classifier, _X_train, _y_train,
                          _X_test, _y_test):
  classifier.fit(_X_train, _y_train)
  score = classifier.score(_X_test, _y_test)
  return print(f"Accuracy: {score:.4f}")

def cv_classification(classifier, _X, _y, nfolds):
  cv_scores = cross_val_score(classifier, _X, _y, cv=nfolds)
  print(f"CV scores:", cv_scores)
  print(f"Mean: {cv_scores.mean()}")


def model_report(classifier, _X_train, _y_train, _X_test, _y_test,
                 encoder, model_name):
  classifier.fit(_X_train, _y_train)
  y_true = _y_test
  y_pred = classifier.predict(_X_test)
  xticks = np.unique(y_pred)
  display = ConfusionMatrixDisplay.from_estimator(
        classifier,
        _X_test,
        _y_test,
        # display_labels=target_names,
        cmap="RdYlGn",
        # cmap="PiYG",
    )
  # ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
  # plt.xticks(rotation=45, ticks=range(0, len(xticks)), labels=xticks)
  # plt.title(f"Model: {model_name}")
  print("Classication report: ")
  print(classification_report(y_true, y_pred))


def cross_validation_accuracy(_classifier, name, _X, _y, scoring='accuracy', **kwargs):
  scores = cross_val_score(_classifier, _X, _y, cv=5, scoring=scoring, **kwargs)
  print(f"Accuracy for classifier: {name}")
  print(f"The cross-validation accuracy mean is: {scores.mean():.3f}")
  print(f"The cross-validation accuracy std is: {scores.std():.3f}")
  print()

